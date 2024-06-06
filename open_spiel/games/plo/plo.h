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

// A generalized version of a Plo poker, a simple but non-trivial poker game
// described in http://poker.cs.ualberta.ca/publications/UAI05.pdf .
//
// Taken verbatim from the linked paper above: "In Plo hold'em, the deck
// consists of two suits with three cards in each suit. There are two rounds.
// In the first round a single private card is dealt to each player. In the
// second round a single board card is revealed. There is a two-bet maximum,
// with raise amounts of 2 and 4 in the first and second round, respectively.
// Both players start the first round with 1 already in the pot.
//
// So the maximin sequence is of the form:
// private card player 0, private card player 1, [bets], public card, [bets]
//
// Parameters:
//     "players"           int    number of players          (default = 2)
//     "action_mapping"    bool   regard all actions as legal and internally
//                                map otherwise illegal actions to check/call
//                                                           (default = false)
//     "suit_isomorphism"  bool   player observations do not distinguish
//                                between cards of different suits with
//                                the same rank              (default = false)

#ifndef OPEN_SPIEL_GAMES_PLO_H_
#define OPEN_SPIEL_GAMES_PLO_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace plo {

class Card{
  public:
    int rank, suit;
    Card(int r, int s){
      rank = r; suit = s;
    }
    Card(std::pair<int, int> rs){
      rank = rs.first; suit = rs.second;
    }
    Card(std::vector<int> rs){
      rank = rs[0]; suit = rs[1];
    }
    bool operator <(const Card& card2) const{
			return rank==card2.rank?suit<card2.suit:rank<card2.rank;
    }
		bool operator ==(const Card& card2) const{
			return rank==card2.rank&&suit==card2.suit;
    }
		bool operator !=(const Card& card2) const{
			return !(*this==card2);
    }
		std::string ToString() const{
			return std::to_string(rank)+"-"+std::to_string(suit);
		}

    friend std::ostream& operator<< (std::ostream& stream, const Card& card) {
      stream << card.ToString();
      return stream;
    }
};

class Hole{
  public:
    std::vector<Card> cards;
    Hole(Card c0, Card c1, Card c2, Card c3){
      cards = {c0, c1, c2, c3};
    }
    Hole(std::vector<Card> cs){
      cards = cs;
    }
		bool operator ==(const Hole& hole2) const{
			return cards==hole2.cards;
    }
		bool operator !=(const Hole& hole2) const{
			return !(*this==hole2);
    }
		std::string ToString() const{
			return "["+cards[0].ToString()+","+cards[1].ToString()+","+cards[2].ToString()+","+cards[3].ToString()+"]";
		}
		void sort(){
			std::sort(cards.begin(), cards.end());
		}
};

class HandScore{
  public:
    std::vector<int> score;
    HandScore(std::vector<int> sc){
      score = sc;
    }
    bool operator ==(const HandScore& hsc2) const{
			return score==hsc2.score;
    }
    bool operator <(const HandScore& hsc2) const{
      for(int i=0;i<(int)score.size();i++){
        if(score[i]==hsc2.score[i]) continue;
        return score[i]<hsc2.score[i];
      }
      return false;
    }
};

// Default parameters.

inline const Card kInvalidCard{-10000, -10000};
inline const Hole kInvalidHole{kInvalidCard, kInvalidCard, kInvalidCard, kInvalidCard};
inline constexpr int kDefaultPlayers = 2;
inline constexpr int kNumSuits = 4;
inline constexpr double kDefaultStacks = 100;
inline constexpr int default_deck_size = 26;
inline const std::vector<double> bet_sizes = {0.33, 0.66, 1};
inline const std::vector<double> raise_sizes = {0.33, 1};

// Number of info states in the 2P game with default params.
inline constexpr int kNumInfoStates = 10000000;

class PloGame;
class PloObserver;

enum ActionType { kF = 0, kX = 1, kC = 2, kB = 3, kR = 4};

class PloState : public State {
 public:
  explicit PloState(std::shared_ptr<const Game> game,
                      bool action_mapping, bool suit_isomorphism);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  // The probability of taking each possible action in a particular info state.
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  // Additional methods
  int round() const { return round_; }
  int deck_remaining() const { return deck_remaining_; }
  std::vector<Card> public_cards() const { return public_cards_; }
  Hole private_hole(Player player) const { return private_hole_[player]; }
  std::vector<Action> LegalActions() const override;

  // Gets the private cards.
  std::vector<Hole> GetPrivateCards() const { return private_hole_; }

  // Gets the public cards.
  std::vector<Card> GetPublicCards() const { return public_cards_; }

  // Gets number of chips in pot.
  int GetPot() const { return pot_; }

  // Gets how much stack each player has.
  std::vector<double> GetStack() const { return stack_; }

  // Gets the action sequence of rounds 1 & 2.
  std::vector<int> GetRound0() const { return round0_sequence_; }
  std::vector<int> GetRound1() const { return round1_sequence_; }

  // Sets the private cards to specific ones. Note that this function does not
  // change the history, so any functions relying on the history will not longer
  // work properly.
  void SetPrivateCards(const std::vector<Hole>& new_private_hole);

  std::vector<int> padded_betting_sequence() const;
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override;

  std::vector<Action> ActionsConsistentWithInformationFrom(
      Action action) const override {
    return {action};
  }

 protected:
  // The meaning of `action_id` varies:
  // - At decision nodes, one of ActionType::{kFold, kCall, kRaise}.
  // - At a chance node, indicates the card to be dealt to the player or
  // revealed publicly. The interpretation of each chance outcome depends on
  // the number of players
  void DoApplyAction(Action move) override;

 private:
  friend class PloObserver;

  int NextPlayer() const;
  void ResolveWinner();
  void NewRound();
  void SequenceAppendMove(int move);
  void Ante(Player player, int amount);
  void SetPrivate(Player player, Action move);
  HandScore GetScoreFrom5(std::vector<Card> cards) const;
  HandScore RankHand(Player player) const;

  // Fields sets to bad/invalid values. Use Game::NewInitialState().
  Player cur_player_;

  int round_;        // Round number (1 or 2).
  int num_winners_;  // Number of winning players.
  int pot_;          // Number of chips in the pot AT THE START OF THE CURRENT ROUND
  std::vector<Card> public_cards_;  // The public card revealed after round 1.
  int deck_remaining_;    // Number of cards remaining; not equal deck_.size()
  int private_hole_dealt_;  // How many private cards currently dealt.
  std::vector<int> players_remaining_;    // Num. players still in (not folded). 0=sb/button heads-up
  std::vector<double> cur_round_bet_; //In the current round of betting, how much each player has bet so far
  double cur_max_bet_; //Max in cur_round_bet_
  bool action_is_closed_; //Whether the action for the current round has closed
  int last_to_act_; //Index of the last player who can act (BB preflop, BU postflop)

  // Is this player a winner? Indexed by pid.
  std::vector<bool> winner_;
  // Each player's hole private cards. Indexed by pid.
  std::vector<Hole> private_hole_;
  // Cards by value (0-6 for standard 2-player game, -1 if no longer in the
  // deck.)
  std::vector<Card> deck_;
  // How much money each player has, indexed by pid.
  std::vector<double> stack_;
  // How much each player has contributed to the pot, indexed by pid.
  std::vector<double> ante_;
  // Sequence of actions for each round. Needed to report information state.
  std::vector<int> round0_sequence_;
  std::vector<int> round1_sequence_;
  // Players cannot distinguish between cards of different suits with the same
  // rank.
  bool suit_isomorphism_;
  bool game_abstraction_;
};

class PloGame : public Game {
 public:
  explicit PloGame(const GameParameters& params);

  int NumDistinctActions() const override { return 5; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override;
  double MaxUtility() const override;
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;

  int MaxGameLength() const override {
    // 2 rounds.
    return 1000;
  }

  int MaxChanceNodesInHistory() const override { return 3; }

  std::string ActionToString(Player player, Action action) const override;
  // New Observation API
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;

  // Used to implement the old observation API.
  std::shared_ptr<PloObserver> default_observer_;
  std::shared_ptr<PloObserver> info_state_observer_;

 private:
  int num_players_;  // Number of players.
  // Players cannot distinguish between cards of different suits with the same
  // rank.
  bool suit_isomorphism_;
  bool game_abstraction_;
};

}  // namespace plo
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PLO_H_
