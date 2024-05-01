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

/*
#include "open_spiel/games/simple_poker/simple_poker.h"

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"
*/

#include "/Users/dnass/dev/open_spiel/open_spiel/games/simple_poker/simple_poker.h"

#include "/Users/dnass/dev/open_spiel/open_spiel/algorithms/get_all_states.h"
#include "/Users/dnass/dev/open_spiel/open_spiel/policy.h"
#include "/Users/dnass/dev/open_spiel/open_spiel/spiel_utils.h"
#include "/Users/dnass/dev/open_spiel/open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace simple_poker {
namespace {

namespace testing = open_spiel::testing;

void BasicSimpleTests() {
  testing::LoadGameTest("simple_poker");
  testing::ChanceOutcomesTest(*LoadGame("simple_poker"));
  testing::RandomSimTest(*LoadGame("simple_poker"), 100);
  testing::RandomSimTestWithUndo(*LoadGame("simple_poker"), 1);
  for (Player players = 2; players <= 2; players++) {
    testing::RandomSimTest(
        *LoadGame("simple_poker", {{"players", GameParameter(players)}}), 100);
  }
  auto observer = LoadGame("simple_poker")
                      ->MakeObserver(kDefaultObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("simple_poker"), observer);
}

void CountStates() {
  std::shared_ptr<const Game> game = LoadGame("simple_poker");
  auto states = algorithms::GetAllStates(*game, /*depth_limit=*/-1,
                                         /*include_terminals=*/true,
                                         /*include_chance_states=*/false);
  // 6 deals * 9 betting sequences (-, p, b, pp, pb, bp, bb, pbp, pbb) = 54
  SPIEL_CHECK_EQ(states.size(), 54);
}

void PolicyTest() {
  using PolicyGenerator = std::function<TabularPolicy(const Game& game)>;
  std::vector<PolicyGenerator> policy_generators = {
      GetAlwaysPassPolicy,
      GetAlwaysBetPolicy,
  };

  std::shared_ptr<const Game> game = LoadGame("simple_poker");
  for (const auto& policy_generator : policy_generators) {
    testing::TestEveryInfostateInPolicy(policy_generator, *game);
    testing::TestPoliciesCanPlay(policy_generator, *game);
  }
}

}  // namespace
}  // namespace simple_poker
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::simple_poker::BasicSimpleTests();
  open_spiel::simple_poker::CountStates();
  open_spiel::simple_poker::PolicyTest();
  open_spiel::testing::CheckChanceOutcomes(*open_spiel::LoadGame(
      "simple_poker", {{"players", open_spiel::GameParameter(3)}}));
  open_spiel::testing::RandomSimTest(*open_spiel::LoadGame("simple_poker"),
                                     /*num_sims=*/10);
  open_spiel::testing::ResampleInfostateTest(
      *open_spiel::LoadGame("simple_poker"),
      /*num_sims=*/10);
}
