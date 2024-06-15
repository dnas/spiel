// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "open_spiel/games/plo/plo.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

#define EPS 0.01
#define PLACEHOLDER -100

namespace open_spiel {
namespace plo {
namespace {

//int counter_zzz = 0;
std::vector<int> kBlinds = {5, 10}; //Button and Big Blind. The button acts first pre-flop, the BB acts first post-flop

const GameType kGameType{/*short_name=*/"plo",
                         /*long_name=*/"Pot Limit Omaha",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"players", GameParameter(kDefaultPlayers)},
                          {"suit_isomorphism", GameParameter(true)},
                          {"game_abstraction", GameParameter(false)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new PloGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::string StatelessActionToString(Action action) {
  if (action == ActionType::kF) {
    return "Fold";
  } else if (action == ActionType::kX) {
    return "Check";
  } else if (action == ActionType::kC) {
    return "Call";
  } else if (action == ActionType::kB) {
    return "Bet";
  } else if (action == ActionType::kR) {
    return "Raise";
  } else {
    SpielFatalError(absl::StrCat("Unknown action: ", action));
  }
  return "Will not return.";
}

RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

// The Observer class is responsible for creating representations of the game
// state for use in learning algorithms. It handles both string and tensor
// representations, and any combination of public information and private
// information (none, observing player only, or all players).
//
// If a perfect recall observation is requested, it must be possible to deduce
// all previous observations for the same information type from the current
// observation.

class PloObserver : public Observer {
 public:
  PloObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  //
  // These helper methods each write a piece of the tensor observation.
  //

  // Identity of the observing player. One-hot vector of size num_players.
  static void WriteObservingPlayer(const PloState& state, int player,
                                   Allocator* allocator) {
    auto out = allocator->Get("player", {state.num_players_});
    out.at(player) = 1;
  }

  // Private card of the observing player. One-hot vector of size num_cards.
  static void WriteSinglePlayerHole(const PloState& state, int player,
                                    Allocator* allocator) {
    auto out = allocator->Get("private_hole", {270725});
    Hole hole_cards = state.private_hole_[player];
    if (hole_cards != kInvalidHole) out.at(hole_cards.cards[0].rank) = 1;
  }

  // Private cards of all players. Tensor of shape [num_players, num_cards].
  static void WriteAllPlayerHoles(const PloState& state,
                                  Allocator* allocator) {
    auto out = allocator->Get("private_hole",
                              {state.num_players_, 270725});
    for (int p = 0; p < state.num_players_; ++p) {
      Hole hole_cards = state.private_hole_[p];
      if (hole_cards != kInvalidHole) out.at(p, state.private_hole_[p].cards[0].rank) = 1;
    }
  }

  // Community cards. One-hot vector of size num_cards.
  static void WriteCommunityCards(const PloState& state,
                                 Allocator* allocator) {
    auto out = allocator->Get("community_card", {270725});
    if (state.public_cards_ != std::vector<Card>{kInvalidCard, kInvalidCard, kInvalidCard}) {
      out.at(state.public_cards_[0].rank, state.public_cards_[1].rank, state.public_cards_[2].rank) = 1;
    }
  }

  // Betting sequence; shape [num_rounds, bets_per_round, num_actions].
  static void WriteBettingSequence(const PloState& state,
                                   Allocator* allocator) {
    const int kNumRounds = small_game?2:4;
    const int kBitsPerAction = 10;
    const int max_bets_per_round = 1000;
    auto out = allocator->Get("betting",
                              {kNumRounds, max_bets_per_round, kBitsPerAction});
    for (int round : {0, 1}) {
      const auto& bets =
          (round == 0) ? state.round0_sequence_ : state.round1_sequence_;
      for (int i = 0; i < bets.size(); ++i) {
        if (bets[i] == ActionType::kC) {
          out.at(round, i, 0) = 1;  // Encode call as 10.
        } else if (bets[i] == ActionType::kR) {
          out.at(round, i, 1) = 1;  // Encode raise as 01.
        }
      }
    }
  }

  // Pot contribution per player (integer per player).
  static void WritePotContribution(const PloState& state,
                                   Allocator* allocator) {
    auto out = allocator->Get("pot_contribution", {state.num_players_});
    for (auto p = Player{0}; p < state.num_players_; p++) {
      out.at(p) = state.ante_[p];
    }
  }

  // Writes the complete observation in tensor form.
  // The supplied allocator is responsible for providing memory to write the
  // observation into.
  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    auto& state = open_spiel::down_cast<const PloState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);

    // Observing player.
    WriteObservingPlayer(state, player, allocator);

    // Private card(s).
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      WriteSinglePlayerHole(state, player, allocator);
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      WriteAllPlayerHoles(state, allocator);
    }

    // Public information.
    if (iig_obs_type_.public_info) {
      WriteCommunityCards(state, allocator);
      iig_obs_type_.perfect_recall ? WriteBettingSequence(state, allocator)
                                   : WritePotContribution(state, allocator);
    }
  }

  // Writes an observation in string form. It would be possible just to
  // turn the tensor observation into a string, but we prefer something
  // somewhat human-readable.

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    auto& state = open_spiel::down_cast<const PloState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);
    std::string result;

    // Private card(s).
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      absl::StrAppend(&result, "[Observer: ", player, "]");
      absl::StrAppend(&result, "[Private: ", state.private_hole_[player].ToString(), "]");
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      absl::StrAppend(
          &result, "[Privates: ", state.private_hole_[0].ToString(), " ", state.private_hole_[1].ToString(), "]");
    }

    // Public info. Not all of this is strictly necessary, but it makes the
    // string easier to understand.
    if (iig_obs_type_.public_info) {
      absl::StrAppend(&result, "[Round ", state.round_, "]");
      absl::StrAppend(&result, "[Player: ", state.cur_player_, "]");
      absl::StrAppend(&result, "[Pot: ", state.pot_, "]");
      absl::StrAppend(&result, "[Stack: ", absl::StrJoin(state.stack_, " "),
                      "]");
      if(small_game) absl::StrAppend(&result, "[Public: ", state.public_cards_[0].ToString(), " ", state.public_cards_[1].ToString(), " ", state.public_cards_[2].ToString(), "]");
      else absl::StrAppend(&result, "[Public: ", state.public_cards_[0].ToString(), " ", state.public_cards_[1].ToString(), " ", state.public_cards_[2].ToString(), " ", state.public_cards_[3].ToString(), " ", state.public_cards_[4].ToString(), "]");
      if (iig_obs_type_.perfect_recall) {
        // Betting Sequence (for the perfect recall case)
        if(small_game) absl::StrAppend(&result, "[Round0: ", absl::StrJoin(state.round0_sequence_, " "),"][Round1: ", absl::StrJoin(state.round1_sequence_, " "), "]");
        else absl::StrAppend(&result, "[Round0: ", absl::StrJoin(state.round0_sequence_, " "),"][Round1: ", absl::StrJoin(state.round1_sequence_, " "),"][Round2: ", absl::StrJoin(state.round2_sequence_, " "),"][Round3: ", absl::StrJoin(state.round3_sequence_, " "), "]");
      } else {
        // Pot contributions (imperfect recall)
        absl::StrAppend(&result, "[Ante: ", absl::StrJoin(state.ante_, " "),
                        "]");
      }
    }

    // Done.
    return result;
  }

 private:
  IIGObservationType iig_obs_type_;
};

PloState::PloState(std::shared_ptr<const Game> game,
                       bool suit_isomorphism, bool game_abstraction)
    : State(game),
      cur_player_(kChancePlayerId),
      round_(0),   // Round number (0 or 1 - preflop or postflop).
      num_winners_(-1),
      pot_(kBlinds[0]+kBlinds[1]),  // Number of chips in the pot.
      action_is_closed_(false),
      last_to_act_(1),
      cur_max_bet_(kBlinds[1]),
      public_cards_(small_game?3:5, kInvalidCard),
      // Number of cards remaining; not equal deck_.size()!
      deck_remaining_(default_deck_size),
      private_hole_dealt_(0),
      players_remaining_(game->NumPlayers()),
      // Is this player a winner? Indexed by pid.
      winner_(game->NumPlayers(), false),
      // Each player's single private card. Indexed by pid.
      private_hole_(game->NumPlayers(), kInvalidHole),
      // How much money each player has, indexed by pid.
      stack_(game->NumPlayers()),
      // How much each player has contributed to the pot, indexed by pid.
      ante_(game->NumPlayers()),
      cur_round_bet_(game->NumPlayers(), 0.0),
      // Sequence of actions for each round. Needed to report information
      // state.
      round0_sequence_(),
      round1_sequence_(),
      round2_sequence_(),
      round3_sequence_(),
      // Players cannot distinguish between cards of different suits with the
      // same rank.
      suit_isomorphism_(suit_isomorphism),
      game_abstraction_(game_abstraction) {
  // Cards by value (0-6 for standard 2-player game, kInvalidCard if no longer
  // in the deck.)
  std::iota(players_remaining_.begin(), players_remaining_.end(), 0);
	for(int p = 0;p<game->NumPlayers();p++){
		stack_[p] = kDefaultStacks-kBlinds[p];
		ante_[p] = kBlinds[p];
    cur_round_bet_[p] = kBlinds[p];
	}
	deck_.clear();
  if(default_deck_size!=26&&default_deck_size!=52) SpielFatalError("Deck size should be 26 or 52");
  for(int rank=0;rank<13;rank++){
    for(int suit=0;suit<default_deck_size/13;suit++){
      deck_.push_back(Card(rank, suit));
    }
  }
  suit_classes_.clear();
  suit_classes_.push_back({0,1,2,3}); //at the start, all suits are equivalent
}

int PloState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return cur_player_;
  }
}

// In a chance node, `move` should be the card to deal to the current
// underlying player.
// On a player node, it should be ActionType::{kF, kX, kB, kC, kR}
void PloState::DoApplyAction(Action move) {
  //std::cout << "-----------------------------------------------------------------" << std::endl;
  //std::cout << "At DoApplyAction, round_ = " << round_ << ", cur_player_ = " << cur_player_ << ", Chance? " << IsChanceNode() << ", move = " << move << std::endl;
  if (IsChanceNode()) {
    SPIEL_CHECK_GE(move, 0);
    SPIEL_CHECK_LT(move, default_deck_size*default_deck_size*default_deck_size*default_deck_size);

    if (private_hole_dealt_ < num_players_) { //round 0 - preflop
      SetPrivate(private_hole_dealt_, move); //since we can only pass Action move to DoApplyAction, we encode the 4 cards in move in base 52
      // When all private cards are dealt, move to player 0 (the Button/Small Blind, who acts first preflop).
      if (private_hole_dealt_ == num_players_) cur_player_ = 0;
    } else if(round_==1) DealPublic(0, 3, move); //flop - 3 cards
    else if(round_==2) DealPublic(3, 1, move); //turn - 1 card
    else if(round_==3) DealPublic(4, 1, move); //river - 1 card
    else SpielFatalError("Round number too large in ChanceNode in DoApplyAction");
  } else {
    SequenceAppendMove(move);
    //Here we encode the action in base 5. 1st digit is action type, 2nd digit is bet sizing index (0 for kF, kX, kC, 0-2 for kB, 0-1 for kR)
    int move_type = move%5;
    int bet_ind = move/5;
    if(move_type == ActionType::kF){
      // Player is now out.
      auto it = std::lower_bound(players_remaining_.begin(), players_remaining_.end(), cur_player_);
      if(it == players_remaining_.end()) SpielFatalError("Couldn't find current player in DoApplyAction.");
      players_remaining_.erase(it);
      action_is_closed_ = true;
      ResolveWinner(); //2 player game - when one folds, it ends;
    }else if(move_type==ActionType::kX){ //checking - just move the game along
      SPIEL_CHECK_EQ(cur_max_bet_, cur_round_bet_[cur_player_]);
      if(round_==0) SPIEL_CHECK_NE(cur_player_, 0); //The button cannot check preflop
      if(cur_player_!=last_to_act_) cur_player_ = NextPlayer();
      else{
        action_is_closed_ = true;
        if(!IsTerminal()) NewRound();
        else ResolveWinner();
      }
    }else if(move_type == ActionType::kC){
      SPIEL_CHECK_GE(cur_max_bet_, 1+cur_round_bet_[cur_player_]);
      int to_call = cur_max_bet_-cur_round_bet_[cur_player_];
      SPIEL_CHECK_GE(stack_[cur_player_], to_call);
      stack_[cur_player_] -= to_call;
      ante_[cur_player_] += to_call;
      cur_round_bet_[cur_player_] += to_call;
      pot_ += to_call;
      action_is_closed_ = true;
      if(round_==0&&cur_player_==0&&cur_max_bet_<=kBlinds[1]) action_is_closed_ = false;

      if (IsTerminal()) ResolveWinner();
      else if(action_is_closed_) NewRound();
      else cur_player_ = NextPlayer();
    }else if(move_type == ActionType::kB){
      SPIEL_CHECK_EQ(cur_max_bet_, 0);
      int to_bet = (int) (bet_sizes[bet_ind]*pot_);
      to_bet = std::max(to_bet, kBlinds[1]); //min bet rule: cannot bet less than 1BB
      to_bet = std::min(to_bet, stack_[cur_player_]); //cannot bet more than stack, takes priority over the min bet rule
      stack_[cur_player_] -= to_bet;
      ante_[cur_player_] += to_bet;
      cur_max_bet_ += to_bet;
      cur_round_bet_[cur_player_] += to_bet;
      pot_ += to_bet;

      if (IsTerminal()) SpielFatalError("Cannot be terminal after a bet");
      else cur_player_ = NextPlayer();
    }else if(move_type==ActionType::kR){
      int to_raise = (int)(cur_max_bet_-cur_round_bet_[cur_player_]+raise_sizes[bet_ind]*(pot_+cur_max_bet_-cur_round_bet_[cur_player_]));
      if(raise_sizes[bet_ind]*(pot_+cur_max_bet_-cur_round_bet_[cur_player_])<cur_max_bet_-cur_round_bet_[cur_player_]) to_raise = 2*(cur_max_bet_-cur_round_bet_[cur_player_]); //min raise rule - must be at least equal to the previous raise
      to_raise = std::min(to_raise, stack_[cur_player_]); //cannot raise more than stack
      stack_[cur_player_] -= to_raise;
      ante_[cur_player_] += to_raise;
      cur_round_bet_[cur_player_] += to_raise;
      cur_max_bet_ = cur_round_bet_[cur_player_];
      pot_ += to_raise;

      if (IsTerminal()) SpielFatalError("Cannot be terminal after a raise");
      else cur_player_ = NextPlayer();
    }else SpielFatalError(absl::StrCat("Move ", move, " is invalid. ChanceNode?", IsChanceNode()));
  }
  /*
  std::cout << "Stacks: [" << stack_[0] << ", " << stack_[1] << "], Pot: " << pot_ << std::endl; 
  for(auto sclass:suit_classes_){
    std::cout << "{";
    for(int suit:sclass) std::cout << suit << ", ";
    std::cout << "}, ";
  }
  std::cout << std::endl;
  */
}

std::vector<std::vector<Card>> PloState::GetIso(std::vector<std::vector<Card>> classes) const{
  //create map from suit class -> counter
  std::map<std::vector<int>, int> class_counter;
  for(auto sclass:suit_classes_) class_counter[sclass] = 0;
  //permutate suits
  for(auto sclass: classes){
    if(sclass.empty()) continue;
    //find the suit in suit_classes_
    for(int i=0;i<(int)suit_classes_.size();i++){
      if(std::lower_bound(suit_classes_[i].begin(), suit_classes_[i].end(), sclass[0].suit)!=suit_classes_[i].end()){
        //convert every card in this class to the suit indicated by suit_classes_[class_counter]
        for(Card& c:sclass) c.suit = suit_classes_[i][class_counter[suit_classes_[i]]];
        class_counter[suit_classes_[i]]++;
        break;
      }
    }
  }
  return classes;
}

int PloState::GetCardIndex(Card c) const{
  int nr_suits = default_deck_size/13;
  return nr_suits*c.rank+c.suit;
}

std::vector<std::vector<Card>> PloState::GetClasses(std::vector<int> inds, bool to_sort) const{
  std::vector<std::vector<Card>> classes(4);
  for(int ind:inds) classes[deck_[ind].suit].push_back(deck_[ind]);
  if(to_sort) std::sort(classes.begin(), classes.end(), comp);
  return classes;
}

void PloState::UpdateSuitClasses(std::vector<int> inds){
  bool equiv[4][4];
  for(int i=0;i<4;i++) for(int j=0;j<4;j++) equiv[i][j] = false;
  for(int i=0;i<4;i++) equiv[i][i] = true;
  for(auto sclass:suit_classes_){
    for(int i=0;i<(int)sclass.size();i++){
      for(int j=i+1;j<(int)sclass.size();j++){
        equiv[i][j] = true;
        equiv[j][i] = true;
      }
    }
  }
  std::vector<std::vector<Card>> classes = GetClasses(inds, false);
  for(int i=0;i<4;i++){
    for(int j=i+1;j<4;j++){
      equiv[i][j] &= comp_eq(classes[i], classes[j]);
      equiv[j][i] = equiv[i][j];
    }
  }
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int k=0;k<4;k++){
        if(equiv[i][j]&&equiv[j][k]&&!equiv[i][k]) SpielFatalError("Suit equivalence is not transitive");
      }
    }
  }
  suit_classes_.clear();
  std::vector<bool> visited(4, false);
  for(int i=0;i<4;i++){
    if(visited[i]) continue;
    std::vector<int> cur_class;
    for(int j=i;j<4;j++){
      if(equiv[i][j]){
        visited[j] = true;
        cur_class.push_back(j);
      }
    }
    suit_classes_.push_back(cur_class);
  }
}

std::vector<Action> PloState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> movelist;
  std::set<Action> moveset;
  if (IsChanceNode()) {
    if(round_==0){ //deal 4 cards to each player
      for(int i=0;i<default_deck_size;i++){
        if(deck_[i] == kInvalidCard) continue;
        for(int j=i+1;j<default_deck_size;j++){
          if(deck_[j] == kInvalidCard) continue;
          for(int k=j+1;k<default_deck_size;k++){
            if(deck_[k] == kInvalidCard) continue;
            for(int l=k+1;l<default_deck_size;l++){
              if(deck_[l] == kInvalidCard) continue;
              if(!suit_isomorphism_) movelist.push_back(i+default_deck_size*j+default_deck_size*default_deck_size*k+default_deck_size*default_deck_size*default_deck_size*l);
              else{
                std::vector<std::vector<Card>> classes = GetClasses({i,j,k,l});
                std::vector<std::vector<Card>> iso = GetIso(classes);
                std::vector<Card> iso_flattened;
                for(auto sclass:iso) for(Card c:sclass) iso_flattened.push_back(c);
                moveset.insert(GetCardIndex(iso_flattened[0])+default_deck_size*GetCardIndex(iso_flattened[1])+default_deck_size*default_deck_size*GetCardIndex(iso_flattened[2])+default_deck_size*default_deck_size*default_deck_size*GetCardIndex(iso_flattened[3]));
              }
            }
          }
        }
      }
    }else if(round_==1){ //deal 3 public cards
      for(int i=0;i<default_deck_size;i++){
        if(deck_[i] == kInvalidCard) continue;
        for(int j=i+1;j<default_deck_size;j++){
          if(deck_[j] == kInvalidCard) continue;
          for(int k=j+1;k<default_deck_size;k++){
            if(deck_[k] == kInvalidCard) continue;
            if(!suit_isomorphism_) movelist.push_back(i+default_deck_size*j+default_deck_size*default_deck_size*k);
            else{
              std::vector<std::vector<Card>> classes = GetClasses({i,j,k});
              std::vector<std::vector<Card>> iso = GetIso(classes);
              std::vector<Card> iso_flattened;
              for(auto sclass:iso) for(Card c:sclass) iso_flattened.push_back(c);
              moveset.insert(GetCardIndex(iso_flattened[0])+default_deck_size*GetCardIndex(iso_flattened[1])+default_deck_size*default_deck_size*GetCardIndex(iso_flattened[2]));
            }
          }
        }
      }
    }else if(round_<=3){ //deal 1 public card
      for(int i=0;i<default_deck_size;i++){
        if(deck_[i] == kInvalidCard) continue;
        if(!suit_isomorphism_) movelist.push_back(i);
        else{
          std::vector<std::vector<Card>> classes = GetClasses({i});
          std::vector<std::vector<Card>> iso = GetIso(classes);
          std::vector<Card> iso_flattened;
          for(auto sclass:iso) for(Card c:sclass) iso_flattened.push_back(c);
          moveset.insert(GetCardIndex(iso_flattened[0]));
        }
      }
    }else SpielFatalError("round_ too big in LegalActions");
    if(suit_isomorphism_) std::vector<Action> movelist(moveset.begin(), moveset.end());
    return movelist;
  }

  // Can't just randomly fold; only allow fold when under pressure.
  if (cur_max_bet_>cur_round_bet_[cur_player_]){
    movelist.push_back(ActionType::kF);
    movelist.push_back(ActionType::kC);
  }
  // Can only chek if the current bet is 0, or if we are the big blind preflop after a limp
  if (cur_max_bet_==cur_round_bet_[cur_player_]) movelist.push_back(ActionType::kX);
  //Can bet postflop if the current bet is 0 and the stack allows
  if(cur_max_bet_==0&&round_>0&&stack_[cur_player_]>0){
    for(int i=0;i<(int)bet_sizes.size();i++){
      movelist.push_back(ActionType::kB+5*i);
      int to_bet = (int) (bet_sizes[i]*pot_);
      if(to_bet>stack_[cur_player_]) break;
    }
  }
  //Can raise if we are preflop, or if cur_max_bet_>cur_round_bet_[cur_player_], and the stack allows
  if((round_==0||cur_max_bet_>cur_round_bet_[cur_player_])&&stack_[cur_player_]>cur_max_bet_-cur_round_bet_[cur_player_]){
    for(int i=0;i<(int)raise_sizes.size();i++){
      movelist.push_back(ActionType::kR+5*i);
      int to_raise = (int)(cur_max_bet_-cur_round_bet_[cur_player_]+raise_sizes[i]*(pot_+cur_max_bet_-cur_round_bet_[cur_player_]));
      if(to_raise>stack_[cur_player_]) break;
    }
  }
  return movelist;
}

void PloState::DealPublic(int strt, int nr_cards, Action move){
  // Encoded in move using base 52 (or 26)
    std::vector<int> inds;
    for(int i=strt;i<strt+nr_cards;i++){
      public_cards_[i] = deck_[move%default_deck_size];
      if(public_cards_[i]==kInvalidCard) SpielFatalError("Trying to deal invalid card");
      inds.push_back(move%default_deck_size);
      deck_remaining_--;
      move/=default_deck_size;
    }
    if(suit_isomorphism_) UpdateSuitClasses(inds);
    for(int i=strt;i<strt+nr_cards;i++){
      deck_[inds[i-strt]] = kInvalidCard;
    }
    // We have finished dealing, let's bet!
    cur_player_ = NextPlayer();
}

std::string PloState::ActionToString(Player player, Action move) const {
  return GetGame()->ActionToString(player, move);
}

std::string PloState::ToString() const {
  std::string result;

  absl::StrAppend(&result, "Round: ", round_, "\nPlayer: ", cur_player_,
                  "\nPot: ", pot_, "\nMoney (p1 p2 ...):");
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, " ", stack_[p]);
  }
  if(small_game) absl::StrAppend(&result, "\nCards (public p1 p2 ...): ", public_cards_[0].ToString(), " ", public_cards_[1].ToString(), " ", public_cards_[2].ToString(), " ");
  else absl::StrAppend(&result, "\nCards (public p1 p2 ...): ", public_cards_[0].ToString(), " ", public_cards_[1].ToString(), " ", public_cards_[2].ToString(), " ", public_cards_[3].ToString(), " ", public_cards_[4].ToString(), " ");
  for (Player player_index = 0; player_index < num_players_; player_index++) {
    absl::StrAppend(&result, private_hole_[player_index].ToString(), " ");
  }

  absl::StrAppend(&result, "\nRound 0 sequence: ");
  for (int i = 0; i < round0_sequence_.size(); ++i) {
    Action action = round0_sequence_[i];
    if (i > 0) absl::StrAppend(&result, ", ");
    absl::StrAppend(&result, StatelessActionToString(action));
  }
  absl::StrAppend(&result, "\nRound 1 sequence: ");
  for (int i = 0; i < round1_sequence_.size(); ++i) {
    Action action = round1_sequence_[i];
    if (i > 0) absl::StrAppend(&result, ", ");
    absl::StrAppend(&result, StatelessActionToString(action));
  }
  if(!small_game){
    absl::StrAppend(&result, "\nRound 2 sequence: ");
    for (int i = 0; i < round2_sequence_.size(); ++i) {
      Action action = round2_sequence_[i];
      if (i > 0) absl::StrAppend(&result, ", ");
      absl::StrAppend(&result, StatelessActionToString(action));
    }
    absl::StrAppend(&result, "\nRound 3 sequence: ");
    for (int i = 0; i < round3_sequence_.size(); ++i) {
      Action action = round3_sequence_[i];
      if (i > 0) absl::StrAppend(&result, ", ");
      absl::StrAppend(&result, StatelessActionToString(action));
    }
  }
  absl::StrAppend(&result, "\n");

  return result;
}

bool PloState::IsTerminal() const {
  int final_round = small_game?1:3;
  return (int)players_remaining_.size() == 1 || (round_ == final_round && action_is_closed_);
}

std::vector<double> PloState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  std::vector<double> returns(num_players_);
  //std::cout << "Returns: [";
  for (auto player = Player{0}; player < num_players_; ++player) {
    // Money vs money at start.
    returns[player] = stack_[player] - kDefaultStacks;
    //std::cout << returns[player] << (player==num_players_-1?"]":", ");
  }
  //std::cout<<std::endl;
  return returns;
}

// Information state is card then bets.
std::string PloState::InformationStateString(Player player) const {
  const PloGame& game = open_spiel::down_cast<const PloGame&>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

// Observation is card then contribution of each players to the pot.
std::string PloState::ObservationString(Player player) const {
  const PloGame& game = open_spiel::down_cast<const PloGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void PloState::InformationStateTensor(Player player,
                                        absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const PloGame& game = open_spiel::down_cast<const PloGame&>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void PloState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const PloGame& game = open_spiel::down_cast<const PloGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> PloState::Clone() const {
  return std::unique_ptr<State>(new PloState(*this));
}

std::vector<std::pair<Action, double>> PloState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;
  std::map<Action, double> outcome_map;
  if(round_==0){ //deal 4 cards to each player
    double nr_holes = ((double)(deck_remaining_*(deck_remaining_-1)*(deck_remaining_-2)*(deck_remaining_-3)))/24.0;
    for(int i=0;i<default_deck_size;i++){
      if(deck_[i] == kInvalidCard) continue;
      for(int j=i+1;j<default_deck_size;j++){
        if(deck_[j] == kInvalidCard) continue;
        for(int k=j+1;k<default_deck_size;k++){
          if(deck_[k] == kInvalidCard) continue;
          for(int l=k+1;l<default_deck_size;l++){
            if(deck_[l] == kInvalidCard) continue;
            if(!suit_isomorphism_) outcomes.push_back({i+default_deck_size*j+default_deck_size*default_deck_size*k+default_deck_size*default_deck_size*default_deck_size*l, 1.0/nr_holes});
            else{
              std::vector<std::vector<Card>> classes = GetClasses({i,j,k,l});
              std::vector<std::vector<Card>> iso = GetIso(classes);
              std::vector<Card> iso_flattened;
              for(auto sclass:iso) for(Card c:sclass) iso_flattened.push_back(c);
              Action to_act = GetCardIndex(iso_flattened[0])+default_deck_size*GetCardIndex(iso_flattened[1])+default_deck_size*default_deck_size*GetCardIndex(iso_flattened[2])+default_deck_size*default_deck_size*default_deck_size*GetCardIndex(iso_flattened[3]);
              if(outcome_map.find(to_act)==outcome_map.end()) outcome_map[to_act] = 1.0;
              else outcome_map[to_act] += 1.0;
            }
          }
        }
      }
    }
  }else if(round_==1){ //deal 3 public cards
    double nr_flops = ((double)(deck_remaining_*(deck_remaining_-1)*(deck_remaining_-2)))/6.0;
    for(int i=0;i<default_deck_size;i++){
      if(deck_[i] == kInvalidCard) continue;
      for(int j=i+1;j<default_deck_size;j++){
        if(deck_[j] == kInvalidCard) continue;
        for(int k=j+1;k<default_deck_size;k++){
          if(deck_[k] == kInvalidCard) continue;
          if(!suit_isomorphism_) outcomes.push_back({i+default_deck_size*j+default_deck_size*default_deck_size*k, 1.0/nr_flops});
          else{
            std::vector<std::vector<Card>> classes = GetClasses({i,j,k});
            std::vector<std::vector<Card>> iso = GetIso(classes);
            std::vector<Card> iso_flattened;
            for(auto sclass:iso) for(Card c:sclass) iso_flattened.push_back(c);
            Action to_act = GetCardIndex(iso_flattened[0])+default_deck_size*GetCardIndex(iso_flattened[1])+default_deck_size*default_deck_size*GetCardIndex(iso_flattened[2]);
            if(outcome_map.find(to_act)==outcome_map.end()) outcome_map[to_act] = 1.0;
            else outcome_map[to_act] += 1.0;
          }
        }
      }
    }
  }else if(round_<=3){
    double nr_turnsorrivers = (double)deck_remaining_;
    for(int i=0;i<default_deck_size;i++){
      if(deck_[i] == kInvalidCard) continue;
      outcomes.push_back({i, 1.0/nr_turnsorrivers});
      if(!suit_isomorphism_) outcomes.push_back({i, 1.0/nr_turnsorrivers});
      else{
        std::vector<std::vector<Card>> classes = GetClasses({i});
        std::vector<std::vector<Card>> iso = GetIso(classes);
        std::vector<Card> iso_flattened;
        for(auto sclass:iso) for(Card c:sclass) iso_flattened.push_back(c);
        Action to_act = GetCardIndex(iso_flattened[0]);
        if(outcome_map.find(to_act)==outcome_map.end()) outcome_map[to_act] = 1.0;
        else outcome_map[to_act] += 1.0;
      }
    }
  }else{
    SpielFatalError("round number too large in ChanceOutcomes");
  }
  if(suit_isomorphism_){
    outcomes.clear();
    double sum_outcomes = 0;
    for(auto outc:outcome_map) sum_outcomes += outc.second;
    for(auto& outc:outcome_map) outc.second/=sum_outcomes;
    for(auto outc:outcome_map) outcomes.push_back({outc.first, outc.second});
  }
  return outcomes;
}

int PloState::NextPlayer() const {
  // If we are on a chance node, it is the first player to play
  if((round_==0&&private_hole_dealt_<num_players_)||(round_==0&&action_is_closed_)) {
    return kChancePlayerId;
  }
  if(cur_player_==kChancePlayerId){
    return (int)(round_>0); //trick: preflop (0) the button (0) acts first, postflop (1) the big blind (1) acts first;
  }
  auto it = std::lower_bound(players_remaining_.begin(), players_remaining_.end(), cur_player_);
  if(it == players_remaining_.end()) SpielFatalError("Could not find player in NextPlayer.");
  if((++it)==players_remaining_.end()) it = players_remaining_.begin();
  return *it;
}

HandScore PloState::GetScoreFrom5(std::vector<Card> cards) const {
  std::sort(cards.begin(), cards.end());
  bool flush = true;
  bool straight = true;
  bool ato5straight = false;
  int trips = -1;
  int pairs = 0;
  int pair_ind = -1;
  //check for straights and flushes
  for(int i=0;i<5;i++){
    if(cards[i].suit!=cards[0].suit) flush = false;
    if(i>0&&cards[i].rank!=cards[i-1].rank+1) straight = false;
    if(i>1&&cards[i].rank==cards[i-1].rank&&cards[i-1].rank==cards[i-2].rank) trips = i;
    if(i>0&&cards[i].rank==cards[i-1].rank){
      pairs++;
      pair_ind = i;
    }
  }
  //check for A-5 straight
  if(cards[0].rank==0&&cards[1].rank==1&&cards[2].rank==2&&cards[3].rank==3&&cards[4].rank==12) ato5straight = true;
  std::vector<int> hand_class(6, -1);
  if((straight||ato5straight)&&flush){ // 9 = straight flush
    hand_class[0] = 9;
    if(straight) hand_class[1] = cards[4].rank;
    else hand_class[1] = cards[3].rank; // in A to 5 straight, highest card is the 5
  }else if(cards[0].rank==cards[3].rank||cards[1].rank==cards[4].rank){ //8 = four of a kind
    hand_class[0] = 8;
    hand_class[1] = cards[2].rank; // get the four of a kind card
    //get the kicker
    if(cards[0].rank==cards[3].rank) hand_class[2] = cards[4].rank;
    else hand_class[2] = cards[0].rank;
  }else if((cards[0].rank==cards[2].rank&&cards[3].rank==cards[4].rank)||(cards[0].rank==cards[1].rank&&cards[2].rank==cards[4].rank)){ // 7 = full house
    hand_class[0] = 7;
    hand_class[1] = cards[2].rank; // get the three of a kind card
    //get the kicker (pair)
    if(cards[0].rank==cards[2].rank&&cards[3].rank==cards[4].rank) hand_class[2] = cards[3].rank;
    else hand_class[2] = cards[0].rank;
  }else if(flush){ // 6 = flush
    hand_class[0] = 6;
    for(int i=4;i>=0;i--) hand_class[5-i] = cards[i].rank; // all cards could be necessary to dispute ties
  }else if(straight||ato5straight){ // 5 = straight
    hand_class[0] = 5;
    if(straight) hand_class[1] = cards[4].rank;
    else hand_class[1] = cards[3].rank; // in A to 5 straight, highest card is the 5
  }else if(trips>=0){ // 4 = three of a kind
    hand_class[0] = 4;
    hand_class[1] = cards[trips].rank;
    int ind_lo = (trips+1)%4, ind_hi = (trips+2)%5;
    if(ind_lo>ind_hi) std::swap(ind_lo, ind_hi);
    hand_class[2] = cards[ind_hi].rank;
    hand_class[3] = cards[ind_lo].rank;
  }else if(pairs>=2){ // 3 = two pairs
    hand_class[0] = 3;
    if(cards[0].rank==cards[1].rank){
      if(cards[2].rank==cards[3].rank){
        hand_class[1] = cards[3].rank; // hi pair
        hand_class[2] = cards[1].rank; // lo pair
        hand_class[3] = cards[4].rank; // kicker
      }else{
        hand_class[1] = cards[4].rank; // hi pair
        hand_class[2] = cards[1].rank; // lo pair
        hand_class[3] = cards[2].rank; // kicker
      }
    }else{
      hand_class[1] = cards[4].rank; // hi pair
      hand_class[2] = cards[2].rank; // lo pair
      hand_class[3] = cards[0].rank; // kicker
    }
  }else if(pairs>0){ // 2 = one pair
    hand_class[0] = 2;
    hand_class[1] = cards[pair_ind].rank;
    std::vector<int> other_inds = {(pair_ind+1)%5, (pair_ind+2)%5, (pair_ind+3)%5};
    std::sort(other_inds.begin(), other_inds.end());
    for(int i=0;i<3;i++) hand_class[2+i] = cards[other_inds[2-i]].rank;
  }else{ // 1 = high card
    hand_class[0] = 1;
    for(int i=4;i>=0;i--) hand_class[5-i] = cards[i].rank; // all cards could be necessary to dispute ties
  }
  return HandScore(hand_class);
}

HandScore PloState::RankHand(Player player) const {
  HandScore max_score = HandScore({-1, -1, -1, -1, -1, -1});
  for(int i=0;i<(int)public_cards_.size();i++){
    for(int j=i+1;j<(int)public_cards_.size();j++){
      for(int k=j+1;k<(int)public_cards_.size();k++){
        for(int l=0;l<4;l++){
          for(int m=l+1;m<4;m++){
            max_score = std::max(max_score, GetScoreFrom5({private_hole_[player].cards[l], private_hole_[player].cards[m], public_cards_[i], public_cards_[j], public_cards_[k]}));
          }
        }
      }
    }
  }
  SPIEL_CHECK_NE(max_score.score[0], -1);
  return max_score;
}

void PloState::ResolveWinner() {
  num_winners_ = kInvalidPlayer;

  if ((int)players_remaining_.size() == 1) {
    winner_[players_remaining_[0]] = true;
    num_winners_ = 1;
    stack_[players_remaining_[0]] += pot_; // += (1-rake_)*pot_, if raked game
    pot_ = 0;
    return;
  } else {
    // Otherwise, showdown!
    // Find the best hand among those still in.
    std::vector<std::pair<HandScore, int>> scores;
    for(int player:players_remaining_) scores.push_back({RankHand(player), player});
    std::sort(scores.begin(), scores.end());
    int j = (int)scores.size()-1;
    num_winners_ = 0;
    while(j>=0&&scores[j].first==scores.back().first){
      winner_[scores[j].second] = true;
      j--;
      num_winners_++;
    }
    // Split the pot among the winners (possibly only one).
    SPIEL_CHECK_TRUE(1 <= num_winners_ && num_winners_ <= num_players_);
    for (Player player_index = 0; player_index < num_players_; player_index++) {
      if (winner_[player_index]) {
        // Give this player their share.
        stack_[player_index] += static_cast<double>(pot_) / num_winners_;
      }
    }
    pot_ = 0;
  }
}

void PloState::NewRound() {
  round_++;
  cur_player_ = kChancePlayerId;
  last_to_act_ = 0;
  cur_max_bet_ = 0;
  action_is_closed_ = false;
  std::fill(cur_round_bet_.begin(), cur_round_bet_.end(), 0);
}

void PloState::SequenceAppendMove(int move) {
  if (round_ == 0) {
    round0_sequence_.push_back(move);
  } else if(round_==1){
    round1_sequence_.push_back(move);
  }else if(round_==2){
    round2_sequence_.push_back(move);
  }else if(round_==3){
    round3_sequence_.push_back(move);
  }else SpielFatalError("SequenceAppendMove: round has to be in [0, 3]");
}

std::vector<int> PloState::padded_betting_sequence() const {
  std::vector<int> history = round0_sequence_;

  // We pad the history to the end of the first round with kPaddingAction.
  history.resize(game_->MaxGameLength() / 2, kInvalidAction);

  // We insert the actions that happened in the second round, and fill to
  // MaxGameLength.
  history.insert(history.end(), round1_sequence_.begin(),
                 round1_sequence_.end());
  history.resize(game_->MaxGameLength(), kInvalidAction);
  return history;
}

void PloState::SetPrivate(Player player, Action move) {
  // Round 1. `move` refers to the encoding of the 4 cards in base 52
  // underlying player (given by `private_hole_dealt_`).
	std::vector<Card> cards_to_give;
  std::vector<int> inds;
	for(int i=0;i<4;i++){
		cards_to_give.push_back(deck_[move%default_deck_size]);
    if(deck_[move%default_deck_size]==kInvalidCard) SpielFatalError("Trying to deal kInvalidCard to player.");
    inds.push_back(move%default_deck_size);
		move/=default_deck_size;
		--deck_remaining_;
	}
	private_hole_[player] = Hole(cards_to_give);
  ++private_hole_dealt_;
  if(suit_isomorphism_) UpdateSuitClasses(inds);
  for(int i=0;i<4;i++){
		deck_[inds[i]] = kInvalidCard;
	}
}

std::unique_ptr<State> PloState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<State> clone = game_->NewInitialState();

  // First, deal out cards:
  Action player_chance = history_.at(player_id).action;
  for (int p = 0; p < GetGame()->NumPlayers(); ++p) {
    if (p == player_id) {
      clone->ApplyAction(history_.at(p).action);
    } else {
      Action chosen_action = player_chance;
      while (chosen_action == player_chance) {
        chosen_action = SampleAction(clone->ChanceOutcomes(), rng()).first;
      }
      clone->ApplyAction(chosen_action);
    }
  }
  for (int action : round0_sequence_) clone->ApplyAction(action);
  if (public_cards_[0] != kInvalidCard && public_cards_[1] != kInvalidCard && public_cards_[2] != kInvalidCard) {
    int ind1, ind2, ind0;
    for(int i=0;i<(int)deck_.size();i++){
      if(deck_[i]==public_cards_[0]) ind0 = i;
      if(deck_[i]==public_cards_[1]) ind1 = i;
      if(deck_[i]==public_cards_[2]) ind2 = i;
    }
    clone->ApplyAction(ind0+ind1*default_deck_size+ind2*default_deck_size*default_deck_size);
    for (int action : round1_sequence_) clone->ApplyAction(action);
    if(!small_game){
      if(public_cards_[3] != kInvalidCard){
        for(int i=0;i<(int)deck_.size();i++){
          if(deck_[i]==public_cards_[3]) ind0 = i;
        }
        clone->ApplyAction(ind0);
        for (int action : round2_sequence_) clone->ApplyAction(action);
        if(public_cards_[4] != kInvalidCard){
          for(int i=0;i<(int)deck_.size();i++){
            if(deck_[i]==public_cards_[4]) ind0 = i;
          }
          clone->ApplyAction(ind0);
          for (int action : round3_sequence_) clone->ApplyAction(action);
        }
      }
    }
  }
  return clone;
}

void PloState::SetPrivateCards(const std::vector<Hole>& new_private_hole) {
  SPIEL_CHECK_EQ(new_private_hole.size(), NumPlayers());
  private_hole_ = new_private_hole;
}

PloGame::PloGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      suit_isomorphism_(ParameterValue<bool>("suit_isomorphism")),
      game_abstraction_(ParameterValue<bool>("game_abstraction")) {
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);
  default_observer_ = std::make_shared<PloObserver>(kDefaultObsType);
  info_state_observer_ = std::make_shared<PloObserver>(kInfoStateObsType);
}

std::unique_ptr<State> PloGame::NewInitialState() const {
  return absl::make_unique<PloState>(shared_from_this(),
                                       /*suit_isomorphism=*/suit_isomorphism_,
                                       /*game_abstraction=*/game_abstraction_);
}

int PloGame::MaxChanceOutcomes() const {
  if (suit_isomorphism_) {
    return 1000000000;
  } else {
    return 1000000000;
  }
}

std::vector<int> PloGame::InformationStateTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (PLACEHOLDER bits each): private card, public card
  // Followed by maximum game length * 2 bits each (call / raise)
  if (suit_isomorphism_) {
    return {(num_players_) + (PLACEHOLDER) + (100 * 2)};
  } else {
    return {(num_players_) + (PLACEHOLDER * 2) + (100 * 2)};
  }
}

std::vector<int> PloGame::ObservationTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (PLACEHOLDER bits each): private card, public card
  // Followed by the contribution of each player to the pot
  if (suit_isomorphism_) {
    return {(num_players_) + (PLACEHOLDER) + (num_players_)};
  } else {
    return {(num_players_) + (PLACEHOLDER * 2) + (num_players_)};
  }
}

double PloGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most a player can win *per opponent* is the most each player can put
  // into the pot, which is the raise amounts on each round times the maximum
  // number raises, plus the original chip they put in to play.
  return (num_players_ - 1);
}

double PloGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most any single player can lose is the maximum number of raises per
  // round times the amounts of each of the raises, plus the original chip
  // they put in to play.
  return -1;
}

std::shared_ptr<Observer> PloGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (params.empty()) {
    return std::make_shared<PloObserver>(
        iig_obs_type.value_or(kDefaultObsType));
  } else {
    return MakeRegisteredObserver(iig_obs_type, params);
  }
}

std::string PloGame::ActionToString(Player player, Action action) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Chance outcome:", action);
  } else {
    return StatelessActionToString(action);
  }
}

}  // namespace plo
}  // namespace open_spiel
