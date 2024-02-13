#include <vector>
#include <algorithm>
#include <fstream>
#include <random>
#include <set>
#include <queue>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <cassert>
#include <map>
using namespace std;
#define dump(x) cerr << #x << " = " << x << endl
#define prl cerr << __LINE__ << " is called" << endl

template<class T>
void debug(T a, T b) {
    for (; a!=b; ++a) cerr << *a << " ";
    cerr << endl;
}

using vi = vector<int>;
using pii = pair<int, int>;

struct Move {
    char s;
    bool inv;
    string toString() const {
        string res;
        if (inv) res+= "-";
        res += s;
        return res;
    }
    Move flip() const {
        Move res = *this;
        res.inv ^= true;
        return res;
    }
    bool operator < (const Move& m) const {
        if (inv != m.inv) return inv < m.inv;
        return s < m.s;
    }
    bool operator == (const Move& m) const {
        return s == m.s && inv == m.inv;
    }
};

vector<Move> parseMove(string s) {
    for (char& c: s) {
        if (c == '.' ) c = ' ';
    }
    stringstream ss;ss << s;
    string token;
    vector<Move> res;
    while (ss >> token) {
        bool inv = false;
        if (token[0] == '-') {
            inv = true;
            token = token.substr(1);
        }
        res.push_back(Move({token[0], inv}));
    }
    return res;
}

struct Perm {
    vi p;
    vector<Move> oper;
    // oper is arranged in ascending order 
    Perm(const vi& a, Move id) {
        p = a;
        oper = {id};
    }
    Perm(const vi& a, vector<Move> oper_) {
        p = a;
        oper = oper_;
    }
    Perm(int n) {
        p.resize(n);
        for (int i = 0; i < n; ++i) p[i] = i;
    }
    Perm() {

    }
    bool exist() const {
        return size() > 0;
    }
    int size() const {
        return p.size();
    }
    int& operator[](std::size_t idx) {
        return p[idx];
    }
    const int& operator[](std::size_t idx) const{
        return p[idx];
    }

    Perm inv() const {
        Perm res(size());
        for (int i = 0; i < size(); ++i) res[p[i]] = i;
        for (int i = (int)oper.size() - 1; i >= 0; --i) res.oper.push_back(oper[i].flip());
        return res;
    }
    Perm operator*(const Perm& b) const {
        // apply self and then b
        Perm res(size());
        for (int i = 0; i < size(); ++i) res[i]=b[p[i]];
        res.oper = oper;
        bool cont = true;
        for (auto x: b.oper) {
            if (cont && !res.oper.empty() && res.oper.back() == x.flip()) {
                res.oper.pop_back();
                continue;
            } else {
                cont = false;
            }
            res.oper.push_back(x);
        }
        return res;
    }
    Perm operator^(int pow) const {
        Perm res(size());
        for (int i = 0; i < pow; ++i) res = res * (*this);
        return res; 
    }

    vi operator +(const vi& a) const {
        // how state changes
        vi res(a.size());
        for (int i = 0; i < a.size(); ++i) res[p[i]] = a[i];
        return res;
    }
    bool isIdentity() const {
        for (int i = 0; i < size(); ++i) if(p[i] != i) return false;
        return true;
    }
    int length() const {
        return oper.size();
    }
};

ostream& operator<<(ostream& os, const Move& m)
{
    os << m.toString();
    return os;
}
ostream& operator<<(ostream& os, const Perm& p)
{
    os << "operation: ";
    for (auto o:p.oper) os << o<<",";
    os << "    permutation: ";
    for (auto pp: p.p) os << pp <<",";

    return os;
}
void checkPerm(vi a) {
    sort(a.begin(), a.end());
    for (int i = 0; i < a.size(); ++i) assert(a[i] == i);
}
void show(const vi& a) {
    debug(a.begin(), a.end());
}

int main(int argc, char** argv) {
    if (true) {
    int N, M, W, n;
    cin >> N >> M >> W;
    n = N;
    string puzzleType;
    cin >> puzzleType;
    vector<string> commands(M);
    for (int i = 0; i < M; ++i) {
        cin>>commands[i];
    }

    vi finalState(N), initialState(N);
    for (int i = 0; i < N; ++i) {
        cin>>finalState[i];
    }
    for (int i = 0; i < N; ++i) {
        cin>>initialState[i];
    }
    // ignore C for now 
    for (int i = 0; i < N; ++i) {
        if (finalState[i] == 2) finalState[i] = 1;
        if (initialState[i] == 2) initialState[i] = 1;
    }

    vector<Perm> gen;
    for (int i = 0; i < M; ++i) {
        vi p(N);
        for (int j = 0; j < N; ++j) {
            cin>>p[j];
        }
        checkPerm(p);
        gen.push_back(Perm(p, parseMove(commands[i])));
    }
    prl;
    assert(N==n);
    map<Move, Perm> m_to_p;
    auto updatePermMoveMap=[&]() {
        m_to_p.clear();
        for (auto g: gen) {
            for (int t = 0; t < 2; ++t) {
                m_to_p[g.oper[0]] = g;
                g = g.inv();
            }
        }
    };
    Perm ans;
    auto l = gen[0], r = gen[1];
    updatePermMoveMap();
    dump(gen[0]);
    dump(gen[0] * gen[1]);
    dump((gen[0] ^ 25) * gen[1]);
    dump((gen[1] ^ 26) * gen[0]);
    show(finalState);
    show(gen[0] + finalState);
    show(gen[1] + finalState);
    
    int n2 = N / 2 + 1;

    vi ringR, ringL;
    int stepR=-1, stepL=-1;
    int Lsize = -1, Cpt = -1;
    {
        for (int i = 0; i < n; ++i) if (gen[0].p[i] == 0) {
            Lsize = i+1;
            break;
        }
        for (int i =0 ; i < n; ++i) {
            if (i > 0 && i != gen[1].p[i]) {
                Cpt = i;
                break;
            }
        }
        for (int i = 0; i < Lsize; ++i) ringL.push_back(i);
        ringR.push_back(0);
        int cpointRIdx=-1;
        for (int i = Lsize; i < n; ++i) {
            ringR.push_back(i);
            if (gen[1].p[i] == Cpt) {
                cpointRIdx = ringR.size();
                ringR.push_back(Cpt);
            }
        }
        stepL = Cpt;
        stepR = ringR.size() - cpointRIdx;
        dump(ringL.size());
        dump(ringR.size());
        dump(stepL);
        dump(stepR);
        assert(ringR.size() == n2);
        assert(ringL.size() == n2);
        dump(Cpt);
        debug(ringL.begin(), ringL.end());
        debug(ringR.begin(), ringR.end());
        for (auto p: ringL) if (p > 0 && p != Cpt) assert(finalState[p] == 0);
        for (auto p: ringR) assert(finalState[p] == 1);
    }
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_int_distribution<> randInt(0, 1000000000);
    auto myrand=[&]() {
        return randInt(engine);
    };
    {
        vector<vi> stepCycleIdx;
        // idx: 0, 26, 52, ...
        // rev: 0, ? ...
        int startIdx = n2 - stepR;
        int curIdx = startIdx;
        vector<bool> marked(n2);
        for (int i = 0; i < ringR.size(); ++i)  if (!marked[i]) {
            vi tmp;
            int curIdx = i;
            while (!marked[curIdx]) {
                marked[curIdx] = true;
                tmp.push_back(curIdx);
                curIdx = (curIdx + stepR) % n2;
            }
            stepCycleIdx.push_back(tmp);
            show(tmp);
        }

        {

            auto calc=[&](vector<bool> setFirst) {
                Perm res = Perm(n);
                vi currentState = initialState;
                vector<vi> currentCycleIdx = stepCycleIdx;
                int lcnt = 0, rcnt = 0;
                auto apply=[&](Perm x) {
                    assert(x.length() == 1);
                    res = res * x;
                    currentState = x + currentState;
                    int diff = 0;
                    if (x.p == r.p) diff = 1;
                    else if (x.p == r.inv().p) diff = -1;
                    for (auto& ve: currentCycleIdx) for (auto& i: ve) {
                        i = (i+n2+diff) % n2;
                    }
                };

                auto rotateUntilCptB=[&]() {
                    bool ok = false;
                    for (int i = 0; i < n2; ++i) {
                        if (currentState[Cpt] == 1) {
                            ok = true;
                            break;
                        }
                        apply(l);
                    }
                    assert(ok);
                };

                int setcnt = 0;
                for (int i = 0; i < n2; ++i) {
                    if (setFirst[i]) {
                        rotateUntilCptB();
                        ++setcnt;
                    }
                    apply(r.inv());
                }

                int curIdx = startIdx; // the index inside r-cycle which comes to Cpt
                int cnt = 0;
                int flip = 0;
                const int L = stepCycleIdx[0].size();
                int wildcard_per_group = (N == 198 ? 1 : 2);
                for (int k = 0; k < stepCycleIdx.size(); ++k) {
                    bool ok = false;
                    auto countZ=[&](int k) {
                        int zcnt = 0;
                        for (int j = 0; j < L; ++j) {
                            if (currentState[ringR[currentCycleIdx[k][j]]] == 0) {
                                ++zcnt;
                            }
                        }
                        return zcnt;
                    };
                    for (int i = 0; i < L; ++i) {
                        if (countZ(k) <= wildcard_per_group) {
                            ok = true;
                            break;
                        }

                        if (currentState[ringR[currentCycleIdx[k][i]]] == 0) {
                            ++flip;
                            while (curIdx != stepCycleIdx[k][i]) {
                                if ((stepCycleIdx[k][i] - curIdx + n2)%n2 <= n2/2) {
                                    apply(r.inv());
                                    curIdx = (curIdx + 1) % n2;
                                } else {
                                    apply(r);
                                    curIdx = (curIdx + n2 - 1) % n2;
                                }
                            }
                            assert(ringR[currentCycleIdx[k][i]] == Cpt);

                            rotateUntilCptB();
                            assert(currentState[ringR[currentCycleIdx[k][i]]] == 1);
                        }
                    }
                    assert(ok);
                }
                int zcnt = 0;
                for (int i = 0; i < n2; ++i) {
                    if (currentState[ringR[i]] == 0) ++zcnt;
                }
                assert(zcnt <= 2);
                return res;
            };
            vector<bool> setFirst(n2);
            for (int trial = 0; trial < 3600 * 2 * 2; ++trial) {
                Perm localBest;
                for (int i = 0; i < n2; ++i) {
                    setFirst[i] = (myrand() % 2 == 0);
                }
                int upditer = -1;
                vector<int> candIdx(n2), initAll(n2);
                for (int i = 0; i < n2; ++i) initAll[i] = i;
                candIdx = initAll;
                shuffle(candIdx.begin(), candIdx.end(), engine);
                for (int iter = 0; iter < 1000; ++iter) {
                    int ridx = candIdx.back();
                    candIdx.pop_back();
                    if (candIdx.empty()) {
                        candIdx = initAll;
                        shuffle(candIdx.begin(), candIdx.end(), engine);
                    }
                    setFirst[ridx] = !setFirst[ridx];
                    auto res = calc(setFirst);
                    if (!localBest.exist() || localBest.length () > res.length()) {
                        localBest = res;
                        upditer = iter;
                        // dump(localBest.length());
                    } else {
                        setFirst[ridx] = !setFirst[ridx];
                    }
                    if (iter - upditer > n2 * 2) break;
                }
                dump(upditer);
                if (!ans.exist() || ans.length() > localBest.length()) {
                    ans = localBest;
                    dump(ans.length());
                }
            }
        }
    }
    dump(ans.length());
    if (true) {
        vi curPerm(n), curState = initialState;
        for (int i = 0; i < n; ++i) curPerm[i] = i;
        for (int i = 0; i < ans.oper.size(); ++i) {
            assert(m_to_p.count(ans.oper[i]));
            curState = m_to_p[ans.oper[i]] + curState;
        }

        debug(curState.begin(), curState.end());
        debug(finalState.begin(), finalState.end());
        int diff = 0;
        for (int i = 0; i < n; ++i) if (curState[i] != finalState[i]) ++diff;
        dump(diff);
        assert(diff <= 4);
    }
    ans = ans.inv();
    reverse(ans.oper.begin(), ans.oper.end());
    string ansString;
    for (int i =0; i < ans.oper.size(); ++i){
        ansString += ans.oper[i].toString();
        if (i + 1 < ans.oper.size()) ansString += '.';
    }
    cout << ansString <<endl;
    }

    return 0;
}