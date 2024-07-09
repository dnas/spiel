#include<iostream>
#include<algorithm>
#include<cstdio>
#include<vector>
#include<cmath>
#include<random>
#include<bitset>
#include<string>
#include<queue>
#include<stack>
#include<set>
#include<map>
#include<deque>
#include<chrono>

std::mt19937 rng(5334);

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

bool comp(std::vector<Card> a, std::vector<Card> b){
  std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
  int i = 0;
  while(i<(int)a.size()&&i<(int)b.size()&&a[i].rank==b[i].rank) i++;
  if(i<(int)a.size()&&i<(int)b.size()) return a[i].rank<b[i].rank;
  return a.size()<b.size();
}

bool comp_eq(std::vector<Card> a, std::vector<Card> b){
  std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
  int i = 0;
  while(i<(int)a.size()&&i<(int)b.size()&&a[i].rank==b[i].rank) i++;
  if(i<(int)a.size()&&i<(int)b.size()) return false;
  return a.size()==b.size();
}

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
    bool operator <(const Hole& hole2) const{
      for(int i=0;i<4;i++) if(cards[i]!=hole2.cards[i]) return cards[i]<hole2.cards[i];
      return false;
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
const int default_deck_size = 52;

//Public information
std::vector<Card> public_cards_;  // The public card revealed after round 1.
//Player information
std::vector<Hole> private_hole_;
// Cards by value (0-6 for standard 2-player game, -1 if no longer in the
// deck.)
std::vector<Card> deck_;

std::vector<std::vector<int>> suit_classes_; //Current vector of classes of indistinguishable suits, plus a counter for each class.
//for example, we might have the pairs {{0,3}, {1,2}}, meaning that suits 0,3 are equivalent, and same for 1,2.
std::vector<std::vector<int>> suit_classes_flop_; //A snapshot of the above vector just before the flop is dealt.
//This is important because the turn may join new classes again

int IndexFromCard(Card c){
  return 4*c.rank+c.suit;
}

Card CardFromIndex(int ind){
  return Card(ind/4, ind%4);
}

std::vector<std::vector<Card>> GetClasses(std::vector<int> inds, bool to_sort = true){
  std::vector<std::vector<Card>> classes(4);
  for(int ind:inds) classes[deck_[ind].suit].push_back(deck_[ind]);
  if(to_sort) std::sort(classes.begin(), classes.end(), comp);
  return classes;
}

void UpdateSuitClasses(std::vector<Card> cs, std::vector<std::vector<int>> my_suit_classes){
  bool equiv[4][4];
  for(int i=0;i<4;i++) for(int j=0;j<4;j++) equiv[i][j] = false;
  for(auto sclass:my_suit_classes){
    for(int i=0;i<(int)sclass.size();i++){
      for(int j=i;j<(int)sclass.size();j++){
        equiv[sclass[i]][sclass[j]] = true;
        equiv[sclass[j]][sclass[i]] = true;
      }
    }
  }

  std::vector<std::vector<Card>> classes(4);
  for(Card c:cs) classes[c.suit].push_back(c);
  
  for(int i=0;i<4;i++){
    for(int j=i+1;j<4;j++){
      //std::cout << "equiv[" << i << "][" << j << "] = " << equiv[i][j] << " comp_eq: " << comp_eq(classes[i], classes[j]) << std::endl;
      equiv[i][j] &= comp_eq(classes[i], classes[j]);
      equiv[j][i] = equiv[i][j];
    }
  }
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int k=0;k<4;k++){
        if(equiv[i][j]&&equiv[j][k]&&!equiv[i][k]){
          std::cerr << "Suit classes are non-transitive" << std::endl;
          exit(1);
        }
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

std::vector<std::vector<Card>> GetIso(std::vector<std::vector<Card>> classes){
  //create map from suit class -> counter
  std::map<std::vector<int>, int> class_counter;
  for(auto sclass:suit_classes_) class_counter[sclass] = 0;
  //permutate suits
  for(auto& sclass: classes){
    if(sclass.empty()) continue;
    //find the suit in suit_classes_
    for(int i=0;i<(int)suit_classes_.size();i++){
      if(std::find(suit_classes_[i].begin(), suit_classes_[i].end(), sclass[0].suit)!=suit_classes_[i].end()){
        //convert every card in this class to the suit indicated by suit_classes_[class_counter]
        for(Card& c:sclass) c.suit = suit_classes_[i][class_counter[suit_classes_[i]]];
        class_counter[suit_classes_[i]]++;
        break;
      }
    }
  }
  return classes;
}

HandScore GetScoreFrom5(std::vector<Card> cards){
  //Returns the hand score of a set of 5 cards. First comes the absolute rank (https://en.wikipedia.org/wiki/List_of_poker_hands#Hand-ranking_categories)
  //Then come the revelant "kicker" information - how to dispute ties in case of same hand rank
  //A HandScore is better (wins) iff its vector is lexicographically greater. See implementation in plo.h
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

HandScore RankHand(int player){
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
  if(max_score.score[0] == -1){
    std::cerr << "max_score is still -1" << std::endl;
    exit(1);
  }
  return max_score;
}

std::vector<std::pair<Hole, double>> ComputeHands(){
  std::vector<std::pair<Hole, double>> holes;
  std::map<Hole, double> outcome_map;
  for(int i=0;i<default_deck_size;i++){
    if(deck_[i] == kInvalidCard) continue;
    for(int j=i+1;j<default_deck_size;j++){
      if(deck_[j] == kInvalidCard) continue;
      for(int k=j+1;k<default_deck_size;k++){
        if(deck_[k] == kInvalidCard) continue;
        for(int l=k+1;l<default_deck_size;l++){
          if(deck_[l] == kInvalidCard) continue;
          std::vector<std::vector<Card>> classes = GetClasses({i,j,k,l});
          std::vector<std::vector<Card>> iso = GetIso(classes);
          std::vector<Card> iso_flattened;
          for(auto sclass:iso) for(Card c:sclass) iso_flattened.push_back(c);
          Hole this_hole = Hole(iso_flattened);
          if(outcome_map.find(this_hole)==outcome_map.end()) outcome_map[this_hole] = 1.0;
          else outcome_map[this_hole] += 1.0;
        }
      }
    }
  }
  double sum_outcomes = 0;
  for(auto outc:outcome_map) sum_outcomes += outc.second;
  for(auto& outc:outcome_map) outc.second/=sum_outcomes;
  for(auto outc:outcome_map) holes.push_back({outc.first, outc.second});
  return holes;
}

double RolloutEquity(int nr_rollouts){
  std::vector<Card> slim_deck;
  for(int i=0;i<default_deck_size;i++) if(deck_[i]!=kInvalidCard) slim_deck.push_back(deck_[i]);
  double equity = 0;
  for(int iii=0;iii<nr_rollouts;iii++){
    std::vector<int> inds_changed(5, -1);
    for(int i=0;i<5;i++){
      while(public_cards_[i]==kInvalidCard){
        int rnd_ind = rng()%((int) slim_deck.size());
        public_cards_[i] = slim_deck[rnd_ind];
        slim_deck[rnd_ind] = kInvalidCard;
        inds_changed[i] = rnd_ind;
      }
    }
    HandScore p0 = RankHand(0); HandScore p1 = RankHand(1);
    if(p1<p0) equity += 1.0;
    else if(p1==p0) equity += 0.5;
    for(int i=0;i<5;i++) slim_deck[inds_changed[i]] = public_cards_[i];
    public_cards_.assign(5, kInvalidCard);
  }
  return equity/nr_rollouts;
}

void AllRolloutEquity(int nr_opphands, int nr_rollouts){
  //Estimates E[HS] (https://poker.cs.ualberta.ca/publications/AAMAS13-abstraction.pdf)
  //using nr_opphands as the number of uniformly sampled opponent hands, and nr_rollouts as the number of rollouts
  std::vector<std::pair<Hole, double>> all_hands = ComputeHands();
  std::reverse(all_hands.begin(), all_hands.end());
  for(auto pair_hd:all_hands){
    Hole cur_hole = pair_hd.first;
    //Update information
    for(int i=0;i<4;i++) deck_[IndexFromCard(cur_hole.cards[i])] = kInvalidCard;
    UpdateSuitClasses(cur_hole.cards, suit_classes_);
    private_hole_[0] = cur_hole;
    //Get rollout equity for this hand
    std::vector<std::pair<Hole, double>> opp_hands = ComputeHands();
    nr_opphands = std::min(nr_opphands, (int)opp_hands.size());
    std::shuffle(opp_hands.begin(), opp_hands.end(), rng);
    double total_equity = 0; double total_prob = 0;
    for(int iii=0;iii<nr_opphands;iii++){
      Hole opp_hole = opp_hands[iii].first;
      for(int i=0;i<4;i++) deck_[IndexFromCard(opp_hole.cards[i])] = kInvalidCard;
      private_hole_[1] = opp_hole;
      total_equity += RolloutEquity(nr_rollouts)*opp_hands[iii].second;
      total_prob += opp_hands[iii].second;

      for(int i=0;i<4;i++) deck_[IndexFromCard(opp_hole.cards[i])] = opp_hole.cards[i];
      private_hole_[1] = kInvalidHole;
    }
    total_equity/=total_prob; //normalization
    std::cout << cur_hole.ToString() << " " << total_equity << std::endl;
    //Go back to previous information
    for(int i=0;i<4;i++) deck_[IndexFromCard(cur_hole.cards[i])] = cur_hole.cards[i];
    suit_classes_.assign(1, {0,1,2,3});
    private_hole_[0] = kInvalidHole;
  }
}

int main(){
  //std::ios::sync_with_stdio(0); std::cin.tie(0); std::cout.tie(0);
	std::cout.precision(6);
  for(int rank=0;rank<13;rank++){
    for(int suit=0;suit<default_deck_size/13;suit++){
      deck_.push_back(Card(rank, suit));
    }
  }
  suit_classes_.clear();
  suit_classes_.assign(1, {0,1,2,3}); //at the start, all suits are equivalent
  private_hole_.assign(2, kInvalidHole);
  public_cards_.assign(5, kInvalidCard);

  AllRolloutEquity(300, 300);
  return 0;
}
