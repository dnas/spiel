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

#include "open_spiel/games/simple_poker/simple_poker.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace simple_poker {
namespace {

// Default parameters.
constexpr int kDefaultPlayers = 2;
constexpr double kAnte = 1;
constexpr int kDefaultCards = 3;

// Facts about the game
const GameType kGameType{/*short_name=*/"simple_poker",
                         /*long_name=*/"Simple Poker",
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
                         {"total_cards", GameParameter(kDefaultCards)}},
                         /*default_loadable=*/true,
                         /*provides_factored_observation_string=*/true,
                        };

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new SimpleGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

open_spiel::RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

class SimpleObserver : public Observer {
 public:
  SimpleObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    const SimpleState& state =
        open_spiel::down_cast<const SimpleState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);
    const int num_players = state.num_players_;

    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      {  // Observing player.
        auto out = allocator->Get("player", {num_players});
        out.at(player) = 1;
      }
      {  // The player's card, if one has been dealt.
        auto out = allocator->Get("private_card", {-1234});
        if (state.history_.size() > player)
          out.at(state.history_[player].action) = 1;
      }
    }

    // Betting sequence.
    if (iig_obs_type_.public_info) {
      if (iig_obs_type_.perfect_recall) {
        auto out = allocator->Get("betting", {2 * num_players - 1, 2});
        for (int i = num_players; i < state.history_.size(); ++i) {
          out.at(i - num_players, state.history_[i].action) = 1;
        }
      } else {
        auto out = allocator->Get("pot_contribution", {num_players});
        for (auto p = Player{0}; p < state.num_players_; p++) {
          out.at(p) = state.ante_[p];
        }
      }
    }
  }

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    const SimpleState& state =
        open_spiel::down_cast<const SimpleState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);
    std::string result;

    // Private card
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      if (iig_obs_type_.perfect_recall || iig_obs_type_.public_info) {
        if (state.history_.size() > player) {
          absl::StrAppend(&result, state.history_[player].action);
        }
      } else {
        if (state.history_.size() == 1 + player) {
          absl::StrAppend(&result, "Received card ",
                          state.history_[player].action);
        }
      }
    }

    // Betting.
    // TODO(author11) Make this more self-consistent.
    if (iig_obs_type_.public_info) {
      if (iig_obs_type_.perfect_recall) {
        // Perfect recall public info.
        for (int i = state.num_players_; i < state.history_.size(); ++i){
          if(state.history_[i].action==ActionType::kPass) result.push_back('p');
          else if(state.history_[i].action==ActionType::kCall) result.push_back('c');
          else result.push_back('r');
        }
      } else {
        // Imperfect recall public info - two different formats.
        if (iig_obs_type_.private_info == PrivateInfoType::kNone) {
          if (state.history_.empty()) {
            absl::StrAppend(&result, "start game");
          } else if (state.history_.size() > state.num_players_) {
            absl::StrAppend(&result,
                            state.history_.back().action ? "Bet" : "Pass");
          }
        } else {
          if (state.history_.size() > player) {
            for (auto p = Player{0}; p < state.num_players_; p++) {
              absl::StrAppend(&result, state.ante_[p]);
            }
          }
        }
      }
    }

    // Fact that we're dealing a card.
    if (iig_obs_type_.public_info &&
        iig_obs_type_.private_info == PrivateInfoType::kNone &&
        !state.history_.empty() &&
        state.history_.size() <= state.num_players_) {
      int currently_dealing_to_player = state.history_.size() - 1;
      absl::StrAppend(&result, "Deal to player ", currently_dealing_to_player);
    }
    return result;
  }

 private:
  IIGObservationType iig_obs_type_;
};

SimpleState::SimpleState(std::shared_ptr<const Game> game, int total_cards)
    : State(game),
      first_bettor_(kInvalidPlayer),
      card_dealt_(total_cards, kInvalidPlayer),
      winner_(kInvalidPlayer),
      showdown_(false),
      remaining_players_(game->NumPlayers()),
      pot_(kAnte * game->NumPlayers()),
      total_cards_(total_cards),
      // How much each player has contributed to the pot, indexed by pid.
      ante_(game->NumPlayers(), kAnte) {}

int SimpleState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return (history_.size() < num_players_) ? kChancePlayerId
                                            : history_.size() % num_players_;
  }
}

void SimpleState::DoApplyAction(Action move) {
  // Additional book-keeping
  if (history_.size() < num_players_) {
    // Give card `move` to player `history_.size()` (CurrentPlayer will return kChancePlayerId, so we use that instead).
    card_dealt_[move] = history_.size();
  } else if (move == ActionType::kRaise) {
    if (first_bettor_ == kInvalidPlayer){
      first_bettor_ = CurrentPlayer();
      pot_ += 1;
      ante_[CurrentPlayer()] += 1;
    }else{
      pot_ += 3;
      ante_[CurrentPlayer()] = 4;
    }
  }else if(move==ActionType::kCall){
    if(CurrentPlayer() == first_bettor_){
      pot_ += 2;
      ante_[CurrentPlayer()] = 4;
    }else{
      pot_ += 1;
      ante_[CurrentPlayer()] += 1;
    }
    showdown_ = true;
  }else if(move==ActionType::kPass){
    if(first_bettor_!=kInvalidPlayer){
      winner_ = (CurrentPlayer()+1)%2;
      remaining_players_ = 1;
      showdown_ = false;
    }
  }

  // We undo that before exiting the method.
  // This is used in `DidBet`.
  history_.push_back({CurrentPlayer(), move});

  // Check for the game being over.
  const int num_actions = history_.size() - num_players_;
  if (first_bettor_ == kInvalidPlayer && num_actions == num_players_) {
    // Nobody bet; the winner is the person with the highest card dealt
    // Losers lose 1, winner wins 1 * (num_players - 1)
    winner_ = kInvalidPlayer;
    int card_ind = total_cards_-1;
    while(winner_==kInvalidPlayer) winner_ = card_dealt_[card_ind--];
    SPIEL_CHECK_NE(winner_, kInvalidPlayer);
  } else if (remaining_players_==1){
    //winner_ = (CurrentPlayer()+1)%2;
    SPIEL_CHECK_NE(winner_, kInvalidPlayer);
    SPIEL_CHECK_NE(winner_, kChancePlayerId);
  }else if(showdown_){
    // There was betting; so the winner is the person with the highest card
    // who stayed in the hand.
    // Check players in turn starting with the highest card.
    winner_ = kInvalidPlayer;
    int card_ind = total_cards_-1;
    while(winner_==kInvalidPlayer) winner_ = card_dealt_[card_ind--];
    SPIEL_CHECK_NE(winner_, kInvalidPlayer);
  }
  history_.pop_back(); //tricky: in spiel.cc, applying an action already adds it to history_. Thus we pop_back to prevent a double add
}

std::vector<Action> SimpleState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) {
    std::vector<Action> actions;
    for (int card = 0; card < card_dealt_.size(); ++card) {
      if (card_dealt_[card] == kInvalidPlayer) actions.push_back(card);
    }
    return actions;
  } else {
    std::vector<Action> actions = {ActionType::kPass};
    if(first_bettor_!=kInvalidPlayer&&ante_[CurrentPlayer()]<ante_[(CurrentPlayer()+1)%2]) actions.push_back(ActionType::kCall);
    if(ante_[CurrentPlayer()]<4&&ante_[(CurrentPlayer()+1)%2]<4) actions.push_back(ActionType::kRaise);
    return actions;
  }
}

std::string SimpleState::ActionToString(Player player, Action move) const {
  if (player == kChancePlayerId)
    return absl::StrCat("Deal:", move);
  else if (move == ActionType::kPass)
    return "Pass";
  else if(move==ActionType::kCall)
    return "Call";
  else return "Raise";
}

std::string SimpleState::ToString() const {
  // The deal: space separated card per player
  std::string str;
  for (int i = 0; i < history_.size() && i < num_players_; ++i) {
    if (!str.empty()) str.push_back(' ');
    absl::StrAppend(&str, history_[i].action);
  }

  // The betting history: p for Pass, c for Call, r for Raise
  if (history_.size() > num_players_) str.push_back(' ');
  for (int i = num_players_; i < history_.size(); ++i) {
    if(history_[i].action==ActionType::kPass) str.push_back('p');
    else if(history_[i].action==ActionType::kCall) str.push_back('c');
    else str.push_back('r');
  }

  return str;
}

bool SimpleState::IsTerminal() const { return winner_ != kInvalidPlayer; }

std::vector<double> SimpleState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  std::vector<double> returns(num_players_);
  for (auto player = Player{0}; player < num_players_; ++player) {
    returns[player] = (player == winner_) ? (pot_ - ante_[player]) : -ante_[player];
    //std::cout << "Player: " << player << " ret: " << returns[player] << " winner? "<< (player == winner_) << std::endl;
    //if(player!=winner_) std::cout << "Winner is actually " << winner_ << std::endl;
  }
  SPIEL_CHECK_FLOAT_NEAR((float)(returns[0]+returns[1]), 0.0, 0.01);
  return returns;
}

std::string SimpleState::InformationStateString(Player player) const {
  const SimpleGame& game = open_spiel::down_cast<const SimpleGame&>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

std::string SimpleState::ObservationString(Player player) const {
  const SimpleGame& game = open_spiel::down_cast<const SimpleGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void SimpleState::InformationStateTensor(Player player,
                                       absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const SimpleGame& game = open_spiel::down_cast<const SimpleGame&>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void SimpleState::ObservationTensor(Player player,
                                  absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const SimpleGame& game = open_spiel::down_cast<const SimpleGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> SimpleState::Clone() const {
  return std::unique_ptr<State>(new SimpleState(*this));
}

/*
void SimpleState::UndoAction(Player player, Action move) {
  if (history_.size() <= num_players_) {
    // Undoing a deal move.
    card_dealt_[move] = kInvalidPlayer;
  } else {
    // Undoing a bet / pass.
    if (move == ActionType::kRaise) {
      pot_ -= 1;
      if (player == first_bettor_) first_bettor_ = kInvalidPlayer;
    }
    winner_ = kInvalidPlayer;
  }
  history_.pop_back();
  --move_number_;
}
*/

std::vector<std::pair<Action, double>> SimpleState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;
  const double p = 1.0 / (total_cards_ - history_.size());
  for (int card = 0; card < card_dealt_.size(); card++) {
    if (card_dealt_[card] == kInvalidPlayer) outcomes.push_back({card, p});
  }
  return outcomes;
}

/*
bool SimpleState::DidBet(Player player) const {
  if (first_bettor_ == kInvalidPlayer) {
    return false;
  } else if (player == first_bettor_) {
    return true;
  } else if (player > first_bettor_) {
    return history_[num_players_ + player].action == ActionType::kRaise;
  } else {
    return history_[num_players_ * 2 + player].action == ActionType::kRaise;
  }
}
*/

/*
std::unique_ptr<State> SimpleState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<State> state = game_->NewInitialState();
  Action player_chance = history_.at(player_id).action;
  for (int p = 0; p < game_->NumPlayers(); ++p) {
    if (p == history_.size()) return state;
    if (p == player_id) {
      state->ApplyAction(player_chance);
    } else {
      Action other_chance = player_chance;
      while (other_chance == player_chance) {
        other_chance = SampleAction(state->ChanceOutcomes(), rng()).first;
      }
      state->ApplyAction(other_chance);
    }
  }
  SPIEL_CHECK_GE(state->CurrentPlayer(), 0);
  if (game_->NumPlayers() == history_.size()) return state;
  for (int i = game_->NumPlayers(); i < history_.size(); ++i) {
    state->ApplyAction(history_.at(i).action);
  }
  return state;
}
*/

SimpleGame::SimpleGame(const GameParameters& params)
    : Game(kGameType, params), num_players_(ParameterValue<int>("players")), total_cards_(ParameterValue<int>("total_cards")) {
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);
  default_observer_ = std::make_shared<SimpleObserver>(kDefaultObsType);
  info_state_observer_ = std::make_shared<SimpleObserver>(kInfoStateObsType);
  private_observer_ = std::make_shared<SimpleObserver>(
      IIGObservationType{/*public_info*/false,
                         /*perfect_recall*/false,
                         /*private_info*/PrivateInfoType::kSinglePlayer});
  public_observer_ = std::make_shared<SimpleObserver>(
      IIGObservationType{/*public_info*/true,
                         /*perfect_recall*/false,
                         /*private_info*/PrivateInfoType::kNone});
}

std::unique_ptr<State> SimpleGame::NewInitialState() const {
  return std::unique_ptr<State>(new SimpleState(shared_from_this(), total_cards_));
}

std::vector<int> SimpleGame::InformationStateTensorShape() const {
  // One-hot for whose turn it is.
  // One-hot encoding for the single private card. (n+1 cards = n+1 bits)
  // Followed by 2 (n - 1 + n) bits for betting sequence (longest sequence:
  // everyone except one player can pass and then everyone can bet/pass).
  // n + n + 1 + 2 (n-1 + n) = 6n - 1.
  return {6 * num_players_ - 1};
}

std::vector<int> SimpleGame::ObservationTensorShape() const {
  // One-hot for whose turn it is.
  // One-hot encoding for the single private card. (n+1 cards = n+1 bits)
  // Followed by the contribution of each player to the pot (n).
  // n + n + 1 + n = 3n + 1.
  return {3 * num_players_ + 1};
}

double SimpleGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end
  // of the game minus then money the player had before starting the game.
  // Everyone puts a chip in at the start, and then they each have one more
  // chip. Most that a player can gain is (#opponents)*2.
  return (num_players_ - 1) * 2;
}

double SimpleGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end
  // of the game minus then money the player had before starting the game.
  // In Simple, the most any one player can lose is the single chip they paid
  // to play and the single chip they paid to raise/call.
  return -2;
}

std::shared_ptr<Observer> SimpleGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (params.empty()) {
    return std::make_shared<SimpleObserver>(
        iig_obs_type.value_or(kDefaultObsType));
  } else {
    return MakeRegisteredObserver(iig_obs_type, params);
  }
}

TabularPolicy GetAlwaysPassPolicy(const Game& game) {
  SPIEL_CHECK_TRUE(
      dynamic_cast<SimpleGame*>(const_cast<Game*>(&game)) != nullptr);
  return GetPrefActionPolicy(game, {ActionType::kPass});
}

TabularPolicy GetAlwaysBetPolicy(const Game& game) {
  SPIEL_CHECK_TRUE(
      dynamic_cast<SimpleGame*>(const_cast<Game*>(&game)) != nullptr);
  return GetPrefActionPolicy(game, {ActionType::kRaise});
}

/*
TabularPolicy GetOptimalPolicy() {
  std::unordered_map<std::string, ActionsAndProbs> policy;

  // All infostates have two actions: Pass (0) and Bet (1).
  // Player 0
  policy["0"] = {{0, 2./3.}, {1, 1./3.}};
  policy["1"] = {{0, 1}, {1, 0}};
  policy["2"] = {{0, 0}, {1, 1}};

  // Player 1
  policy["0b"] = {{0, 1}, {1, 0}};
  policy["1b"] = {{0, 1./ 2.}, {1, 1. / 2.}};
  policy["2b"] = {{0, 0}, {1, 1}};
  return TabularPolicy(policy);
}
*/
}  // namespace simple_poker
}  // namespace open_spiel
