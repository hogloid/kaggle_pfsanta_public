#include <vector>
#include <algorithm>
#include <fstream>
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
    string s;
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
        res.push_back(Move({token, inv}));
    }
    return res;
}

struct Perm {
    vi p;
    vector<Move> oper;
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
        // apply b first and then *this
        Perm res(size());
        for (int i = 0; i < size(); ++i) res[i]=p[b[i]];
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
    vi operator *(const vi& a) const {
        // where each element of permutation is permuted to
        vi res(a.size());
        for (int i = 0; i < a.size(); ++i) res[i] = p[a[i]];
        return res;
    }
    vector<vi> operator *(const vector<vi>& a) const {
        // where each element of permutation is permuted to
        vector<vi> res(a.size());
        for (int i = 0; i < a.size(); ++i) {
            for (auto e: a[i]) {
                res[i].push_back(p[e]);
            }
        }
        return res;
    }
    vi operator +(const vi& a) const {
        // how state changes
        vi res(a.size());
        for (int i = 0; i < a.size(); ++i) res[i] = a[p[i]];
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

struct Cell{
    Perm p;
    bool isNew;
};

using Table = vector<vector<Cell> >;
string tablePath;

Table loadTable(string puzzleType, int n, bool dry) {
    Table table(n, vector<Cell>(n));
    FILE *fp = fopen(tablePath.c_str(), "r");
    if (dry || fp == NULL) {
        for (int i = 0; i < n; ++i) {
            table[i][i] = {Perm(n), true};
        }
        return table;
    }
    dump("loading");
    dump(puzzleType);
    int cnt;
    fscanf(fp, "%d", &cnt);
    dump(cnt);
    for (int x = 0; x < cnt; ++x) {
        int i, j;
        fscanf(fp, "%d %d", &i, &j);
        assert(i < n && j < n && i >= 0 && j >= 0);
        auto& elem = table[i][j];
        assert(!elem.p.exist());
        int opLength;
        fscanf(fp, "%d", &opLength);
        for (int i = 0; i < opLength; ++i) {
            char oper[1000];
            int inv;
            fscanf(fp, "%s %d", oper, &inv);
            elem.p.oper.push_back(Move{oper, (bool)inv});
        }
        int permLength;
        fscanf(fp, "%d",  &permLength);
        assert(permLength == n);
        for (int i = 0; i < permLength; ++i) {
            int pp;
            fscanf(fp, "%d", &pp);
            elem.p.p.push_back(pp);
        }
        elem.isNew = false;
    }
    fclose(fp);
    prl;
    return table;
}
void serializeTable(Table& table, string puzzleType) {
    FILE *fp = fopen(tablePath.c_str(), "w");

    int n = table.size(), cnt = 0;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        auto& elem = table[i][j];
        if (!elem.p.exist()) continue;
        ++cnt;
    }
    fprintf(fp, "%d\n", cnt);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        auto& elem = table[i][j];
        if (!elem.p.exist()) continue;
        fprintf(fp, "%d %d\n", i, j);
        fprintf(fp, "%d ", elem.p.length());
        for (auto o: elem.p.oper) {
            fprintf(fp, "%s %d ", o.s.c_str(), (int)o.inv);
        }
        fprintf(fp, "\n");

        fprintf(fp, "%d ", elem.p.size());
        for (auto pp: elem.p.p) {
            fprintf(fp, "%d ", pp);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int calcTargetCnt(string puzzleType, int n) {
    for (auto &c: puzzleType) {
        if (c == '/' || c == '_') c = ' ';
    }
    stringstream ss;
    ss << puzzleType;
    string type;
    ss >> type;
    if (type == "wreath") {
        int a, b;
        ss >> a >> b;
        int n = a + b - 2;
        return n * (n + 1) / 2;
    } else if (type == "globe") {
        int a, b;
        ss >> a >> b;
        int H = a + 1, W = b * 2;
        int result = 0;
        if (H%2) {
            result += W * 2 - 1;
            --H;
        }
        result += (2*W+1)*2*W/2*H/2;
        return result;
    } else if (type == "cube") {
        int a;
        ss>>a;
        int result = 0;
        if (a&1) {
            result += 7*6/2;
        }
        result += 25*24/2*(a*a/4);
        return result;
    } else if (type.substr(0, 3) == "swd") {
        return 25*24/2 * (n / 24);
    }
    assert(false);
}

Perm updateOne(const vector<Perm>& gen, Table& table, vi& base, int i, const Perm& p) {
    int j = p.p[base[i]];
    Cell p_inv = {p.inv(), true};
    Perm result;
    if (table[i][j].p.exist()){
        result = table[i][j].p * p;
        if (p.length() < table[i][j].p.length()) {
            table[i][j] = p_inv;
            updateOne(gen, table, base, i, p_inv.p); // necessary?
        }
    } else {
        table[i][j] = p_inv;
        updateOne(gen, table, base, i, p_inv.p); // ?
        result = Perm(p.size());
    }
    return result;
}

void updateAll(const vector<Perm>& gen, Table& table, vi& base, int wordLimit, int startRow, Perm p) {
    int i = startRow;
    while (i < table.size() && !p.isIdentity() && p.length() < wordLimit) {
        p = updateOne(gen, table, base, i, p);
        ++i;
    }
}

int synthesis(Table& table, const vector<Perm>& gen, vi& base, double lim) {
    int result = 0;
    deque<pii> psNew, psAll;
    for (int i = 0; i < base.size(); ++i) {
        for (int x = 0; x < table[i].size(); ++x) {
            if (!table[i][x].p.exist()) continue;
            psAll.push_back({i, x});
            if (table[i][x].isNew) psNew.push_back({i, x});
        }
        for (pii x : psNew) {
            for (pii y : psAll) {
                if (x > y && table[x.first][x.second].isNew && table[y.first][y.second].isNew) continue;
                ++result;
                updateAll(gen, table, base, (int)lim, min(x.first, y.first), table[x.first][x.second].p * table[y.first][y.second].p);
            }
        }
        const int L = 2;
        while (!psNew.empty() && i-psNew[0].first >= L) {
            table[psNew[0].first][psNew[0].second].isNew = false;
            psNew.pop_front();
        }
        while (!psAll.empty() && i-psAll[0].first >= L) psAll.pop_front();
    }
    for (int i = 0; i < base.size(); ++i) {
        for (int x = 0; x < table[i].size(); ++x) table[i][x].isNew = false;
    }
    return result;
}

int fillOrbits(Table& table, double lim) {
    int result = 0;
    const int m = table.size(), n = table[0].size();
    for (int i = 0; i < m; ++i) {
        set<int> orbit;
        vector<bool> orbit_ar(n);
        for (int x = 0; x < n; ++x) {
            if (table[i][x].p.exist()) {
                orbit.insert(x);
                orbit_ar[x] = true;
            }
        }
        for (int j = i + 1; j < m; ++j) {
            for (int x = 0; x < n; ++x) {
                if (table[j][x].p.exist()) {
                    auto xinv = table[j][x].p.inv();
                    vi orbit_x = table[j][x].p * vi(orbit.begin(), orbit.end());
                    for (auto pt: orbit_x) { // choose from orbit_x OR, !orbit_ar
                        if (!orbit_ar[pt] && table[i][xinv.p[pt]].p.exist()) {
                            auto pNew = table[i][xinv.p[pt]].p * xinv;
                            if (pNew.length() < lim) {
                                table[i][pt] = {pNew, true};
                                ++result;
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}
void createTable(Table& table, vector<Perm> gen, vi& base, int maxRound, int refreshRound, int wordLimit, int targetCnt=-1, bool enableOrbits = true, bool enableSynthesis = true, bool cubeValidityCheck = true) {
    prl;
    const int n = gen[0].size(), m = base.size();
    if (targetCnt == -1) {
        targetCnt = n * (n-1)/2;
    }
    dump(targetCnt);
    // vector<vector<Cell> > table(m, vector<Cell>(n));
    // for (int i = 0; i < m; ++i) table[i][base[i]] = {Perm(n), true};
    
    double lim = wordLimit;
    const int k = gen.size();
    int cnt = 0;
    dump(gen.size());
    auto cubeValidityDiff=[&](const Perm& cur) {
        // update every time front rotation occurs
        map<string, int> sum;
        for (const auto& o: cur.oper) {
            sum[o.s] += o.inv ? -1: 1;
        }
        bool ng = false;
        int result = 0;
        for (auto e: sum) {
            if (e.first[0] == 'f') { // this is special logic for cube!
                result += (e.second % 2 == 0? 0:1);
            } else {
                result += abs(e.second);
            }
        }
        return result;
    };
    function<bool(const Perm&, int)> dfs;
    dfs=[&](const Perm& cur, int remDepth) {
        assert(remDepth >= 0);
        if (cubeValidityCheck) {
            if (cubeValidityDiff(cur) > remDepth) {
                return false;
            }
        }
        if (remDepth > 0) {
            for (const auto &g: gen) {
                auto nxt = g * cur;
                if (nxt.length() > cur.length()) {
                    if (dfs(nxt, remDepth - 1)) {
                        return true;
                    }
                }
                auto invg = g.inv();
                if (invg.p == g.p) continue;
                nxt = invg * cur;
                if (nxt.length() > cur.length()) {
                    if (dfs(nxt, remDepth - 1)) {
                        return true;
                    }
                }
            }
        } else {
            if (cubeValidityCheck) {
                assert(cur.length() % 2 == 0);
            }
            ++cnt;

            if (cnt >= maxRound) return true;
            updateAll(gen, table, base, (int)lim, 0, cur);
            if (cnt % refreshRound == 0) {
                prl;

                int filledCnt = 0, tot = 0;
                double meanCost = 0.0;
                for (int i = 0; i < m; ++i) {
                    int tcnt = 0;
                    double tsum = 0;
                    for (int j = 0; j < table[i].size(); ++j) {
                        if (!table[i][j].p.exist()) continue;
                        ++filledCnt;
                        ++tcnt;
                        tsum+=table[i][j].p.length();
                        tot+=table[i][j].p.length();
                    }
                    if (tcnt > 0)
                        meanCost += tsum / tcnt;
                }
                dump(m);
                dump(cnt);
                dump(filledCnt);dump(meanCost);
                dump(lim);
                dump(cur.length());
                if (filledCnt >= targetCnt) return true;
                
                if (enableSynthesis) {
                    dump(synthesis(table, gen, base, lim));
                }
                if (enableOrbits) {
                    dump(fillOrbits(table, lim));
                }
                if (lim < 1e3) {
                    lim += 20;
                }
            }
        }
        return false;
    };
    for (int depth = 0; ; ++depth) {
        dump(depth);
        if (dfs(Perm(n), depth)) {
            break;
        }
    }
}

Perm factorize(const vi& base, const Table& table, vector<vi> target, int K, vi sampleTarget, bool sampleCheck, vector<Perm> gen) {
    sort(gen.begin(), gen.end(), [](const Perm& a, const Perm& b) {
        return a.length() < b.length();
    });
    int n = target.size();
    vector<Perm> initials = {Perm(n)};
    // for (int i = 0; i < min(10, (int)gen.size()); ++i) {
    //     initials.push_back(gen[i]);
    // }
    Perm ans;
    int cost = 1e9;

    for (auto initial: initials) {
        auto valid=[&](vi sampleTarget, int from) {
            if (!sampleCheck) return true;
            for (int i = from; i < base.size(); ++i) {
                int to = sampleTarget[i];
                if (table[i][to].p.exist()) {
                    sampleTarget = table[i][to].p * sampleTarget;
                } else {
                    return false;
                }
            }
            for (int i = 0; i < n; ++i) if (sampleTarget[i] != i) {
                return false;
            }
            return true;
        };
        struct State {
            Perm p;
            vector<vi> target;
            vi sampleTarget;
            bool isValid;
        };
        assert(valid(initial * sampleTarget, 0));
        vector<State> cands = {{Perm(n), initial * target, initial * sampleTarget, true}}, next;
        for (int i = 0; i < base.size(); ++i) {
            for (auto &e: cands) {
                auto [p, target, sampleTarget, isValid] = e;
                for (int& e:target[i]) {
                    if (table[i][e].p.exist()) {
                        // if (e != sampleTarget[i]) continue;

                        auto nextTarget = sampleTarget;
                        for (int j = i+1; j < n; ++j) if (sampleTarget[j] == e) {
                            swap(nextTarget[i], nextTarget[j]);
                            break;
                        }
                        nextTarget = table[i][e].p * nextTarget;
                        assert(nextTarget[i] == i);

                        next.push_back({table[i][e].p * p, table[i][e].p * target, nextTarget, valid(nextTarget, i+1)});
                    }
                }
            }
            if (next.empty()) {
                dump("missing");
                dump(i);
                return cands[0].p.inv();
            }
            sort(next.begin(), next.end(), [](const auto& a, const auto& b) {
                return a.p.length() < b.p.length();
            });
            dump(i);
            dump(next[0].p.length());
            if (next.size() > K) {
                // sort(next.begin() + K / 2, next.end(),  [](const auto& a, const auto& b) {
                //     if (a.isValid != b.isValid) {
                //         return a.isValid > b.isValid;
                //     }
                //     return a.p.length() < b.p.length();
                // });
                next.resize(K);
            }
            cands.swap(next);
            next.clear();
        }
        auto res = initial * cands[0].p.inv();
        if (cost > res.length()) {
            cost = res.length();
            ans = res;
        }
    }
    // for (int i = 0; i < n; ++i) if (target[i] != i) {
    //     return Perm();
    // }
    return ans;
}

vector<pair<vi, vi> > createGroups(vi initialState, vi finalState, Table& table, string puzzleType, Perm reOrder) {
    auto reOrdInv = reOrder.inv();
    vector<pair<vi, vi> > groups;
    const int N = initialState.size();
    initialState = reOrdInv + initialState;
    finalState = reOrdInv + finalState;
    if (puzzleType[0] == 'g') {
        int a, b, H, W;
        sscanf(puzzleType.c_str(), "globe_%d/%d", &a, &b);
        H = a + 1;
        W = b * 2;

        map<pii, vi> mp1, mp2;
        for (int i = 0; i < N; ++i) {
            int h = i / W;
            h = min(h, H-1-h);
            mp1[{h, initialState[i]}].push_back(reOrdInv.p[i]);
            mp2[{h, finalState[i]}].push_back(reOrdInv.p[i]);
        }
        for (auto& e: mp1) {
            auto [key, list] = e;
            auto list2 = mp2[key];
            assert(list.size() == list2.size());
            groups.push_back({list, list2});
        }
    } else if (puzzleType[0] == 'c') {
        int a;
        sscanf(puzzleType.c_str(), "cube_%d", &a);

        map<tuple<int, int, int>, vi> mp1, mp2;
        for (int i = 0; i < N; ++i) {
            int face = a*a;
            int w = (i%face)%a, h = (i%face)/a, w2 = w, h2 = h;
            for (int t = 0; t < 4; ++t) {
                auto w3 = a-1-h, h3 = w;
                h = h3; w = w3;
                if (make_pair(h2, w2) > make_pair(h, w)) {
                    h2 = h;
                    w2 = w;
                }
            }
            mp1[{h2, w2, initialState[i]}].push_back(reOrdInv.p[i]);
            mp2[{h2, w2, finalState[i]}].push_back(reOrdInv.p[i]);
        }
        for (auto& e: mp1) {
            auto [key, list] = e;
            auto list2 = mp2[key];
            assert(list.size() == list2.size());
            groups.push_back({list, list2});
        }
    } else {
        map<int, vi> mp1, mp2;
        for (int i = 0; i < N; ++i) {
            mp1[initialState[i]].push_back(reOrdInv.p[i]);
            mp2[finalState[i]].push_back(reOrdInv.p[i]);
        }
        for (auto& e: mp1) {
            auto [key, list] = e;
            auto list2 = mp2[key];
            assert(list.size() == list2.size());
            groups.push_back({list, list2});
        }
    }
    return groups;

    // vi transPerm(N);
    // for (auto e: groups){
    //     auto [A, B] = e;
    //     for (int i = 0; i < A.size(); ++i) {
    //         transPerm[B[i]] = A[i];
    //     }
    // }
    // return transPerm;
}

Perm generateOrdering(const string puzzleType, int N, vector<bool> mismatch) {
    vi perm;
    if (puzzleType[0] == 'c') {
        int s;
        sscanf(puzzleType.c_str(), "cube_%d", &s);
        int half = (s+1) / 2, half2 = s / 2;
        map<tuple<int, int, bool>, vector<int> > series;
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                for (int f = 0 ; f < 6; ++f) {
                    int id = (i * s + j) + f * s * s;
                    int a = i, b = j, flag = false;
                    if (a > s-1-a) a = s-1-a, flag^=true;
                    if (b > s-1-b) b = s-1-b, flag^=true;
                    if (a > b) swap(a, b), flag ^= true;
                    series[make_tuple(a,b, flag)].push_back(id);
                }
            }
        }
        for (auto e: series) {
            auto [a, b, f] = e.first;
            if (a == 0) {
                perm.insert(perm.end(), e.second.begin(), e.second.end());
            }
        }
        for (auto e: series) {
            auto [a, b, f] = e.first;
            if (a > 0 && a!=b) {
                perm.insert(perm.end(), e.second.begin(), e.second.end());
            }
        }
        for (auto e: series) {
            auto [a, b, f] = e.first;
            if (a > 0 && a==b) {
                perm.insert(perm.end(), e.second.begin(), e.second.end());
            }
        }
        dump(perm.size());
        assert(perm.size() == N);
    } else if (puzzleType[0] == 'g') {
        int a, b;
        sscanf(puzzleType.c_str(), "globe_%d/%d", &a, &b);
        int H = a + 1, W = b * 2;
        for (int i = 0; i < (H+1)/2; ++i) {
            for (int j = 0; j < W; ++j) {
                perm.push_back(i * W + j);
                if (i != H-1-i) perm.push_back((H-1-i)*W+j);
            }
        }
        dump(perm.size());
        assert(perm.size() == N);
    } else {
        for (int i = 0; i < N; ++i) {
            perm.push_back(i);
        }
    }
    vi rev(N);
    for (int i = 0; i < N; ++i) rev[perm[i]] = i;

    vector<pair<bool, pii> > order2;
    dump("foo");
    debug(perm.begin(), perm.end());
    debug(mismatch.begin(), mismatch.end());
    for (int i = 0; i < N; ++i) order2.push_back({mismatch[perm[i]], {i, perm[i]}});
    sort(order2.begin(), order2.end());
    vi perm2(N);
    for (int i = 0; i < N; ++i) perm2[i] = order2[i].second.second;
    debug(perm2.begin(), perm2.end());
    Perm p;
    p.p = perm2;
    return p;
}

pair<string, vector<int> > parsePuzzleType(const string s) {
    if (s[0] == 'g') {
        int a, b;
        sscanf(s.c_str(), "globe_%d/%d", &a, &b);
        return {"globe", {a+1, b*2}};
    }
    assert(false);
}

void generatePreFoundCombo(vector<Perm>& gen, string puzzleType) {
    if (puzzleType[0] == 'g') {
        int a, b;
        sscanf(puzzleType.c_str(), "globe_%d/%d", &a, &b);
        int H = a+1, W = b*2;
        vector<Perm> combo, combo_deriv;
        for (int i = 0; i < H / 2; ++i) {
            for (int j = 0; j < W; ++j) {
                auto slice = gen[H+j];
                assert(slice.oper[0].s[0]=='f');
                auto r1 = gen[H-1-i].inv();
                auto r2 = r1.inv();
                auto r3 = gen[i].inv();
                auto r4 = r3.inv();
                assert(r1.oper[0].s[0] == 'r');
                assert(r3.oper[0].s[0] == 'r');

                auto combo_type2 = r4*slice*r3*r2*slice*r1;
                // gen.push_back(combo_type2);
                auto combo_type3 = slice*r4*slice*r3*r2*slice*r1*slice;
                // gen.push_back(combo_type3);

                auto combo_type1 = r1*slice*r2;
                auto combo_type4 = slice*r2*slice*r4*slice*r3*slice*r1;
                // dump(combo_type1);
                combo.push_back(combo_type1);
                combo.push_back(combo_type2);
                combo.push_back(combo_type3);
                combo.push_back(combo_type4);
            }
        }
        // sort(combo.begin(), combo.end(), [](const auto& a, const auto& b) {return a.length() < b.length(); });
        // string fn = puzzleType;
        // for (auto& c: puzzleType) {
        //     if (c == '/') c = '_';
        // }
        // fn = (string)"../combo/" + puzzleType + ".txt";
        // dump(fn);
        // freopen(fn.c_str(), "w+", stdout);
        // for (auto& p: combo) {
        //     string s;
        //     for (auto m: p.oper) {
        //         s += m.toString();
        //         s += ",";
        //     }
        //     s = s.substr(0, s.size() - 1);
        //     printf("%s\n", s.c_str());
        // }
        // exit(0);
        for (auto c:combo) {
            for (int i = 0; i < H; ++i) {
                auto row = gen[i];
                combo_deriv.push_back(row * c * row.inv());
                combo_deriv.push_back(row.inv() * c * row);
            }
            for (int j = 0; j < W; ++j) {
                auto slice = gen[H+j];
                combo_deriv.push_back(slice * c);
                combo_deriv.push_back(c * slice);
            }
        }
        for (auto c: combo) gen.push_back(c);
        for (auto c: combo_deriv) gen.push_back(c);

        if (H % 2 == 1) { // only for H is odd!
            for (int j = 0; j < H; ++j) {
                auto x = gen[j].inv(), y = x.inv();
                Perm X = x, Y = y;
                for (int i = 0; i < W / 2; ++i) {
                    gen.push_back(X);
                    gen.push_back(Y);
                    X = X * x;
                    Y = Y * y;
                }
            }
        }
    }
}
void checkPerm(vi a) {
    sort(a.begin(), a.end());
    for (int i = 0; i < a.size(); ++i) assert(a[i] == i);
}

vi generateSampleTarget(string commands, map<Move, Perm> m_to_p, int n) {
    vector<Move> ms = parseMove(commands);
    vi ar(n);
    for (int i = 0; i < n; ++i) ar[i] = i;
    for (auto m:ms) {
        assert(m_to_p.count(m) > 0);
        ar = m_to_p[m] + ar;
    }
    return ar;
}

Perm createInitialSolution(const vi& initial, const vi& final, vector<Perm> gen, const string puzzleType) {
    dump("create initial...");
    dump(gen.size());
    sort(gen.begin(), gen.end(), [&](const Perm& a, const Perm& b) {
        return a.length() < b.length();
    });
    const int n = initial.size();
    vector<Perm> ps;
    auto puzzleInfo = parsePuzzleType(puzzleType);
    assert(puzzleInfo.first == "globe");
    int H = puzzleInfo.second[0], W = puzzleInfo.second[1];
    auto eval=[&](const Perm& a) {
        //specialized for cube
        vi after = a + initial;
        int res = 0;
        auto countSet=[&](const vi& state) {
            map<pii, int> S;
            for (int i = 0; i < n; i += 2) {
                int a = state[i], b = state[i+1];
                if (a < b) swap(a, b);
                S[{a, b}] += 1;
            }
            return S;
        };
        int result = 0;
        auto afterS = countSet(after), finalS = countSet(final);
        for (auto e: afterS) {
            result += min(e.second, finalS[e.first]);
        }
        return result;
    };
    auto tot=[n](const vector<Perm>& x) {
        Perm res(n);
        for (const auto& p: x) res = p * res;
        return res;
    };
    auto eval2=[&](const vector<Perm>& x) {
        return eval(tot(x));
    };
    dump(eval2({}));
    while (true) {
        const int m = ps.size();
        int best_score = eval2(ps);
        if (best_score >= W * (H/2)) {
            break;
        }
        vector<Perm> nxt;
        vector<Perm> sfx, pfx;
        Perm cur(n);
        for (int i = 0; i < m; ++i) {
            pfx.push_back(cur);
            cur = ps[i] * cur;
        }
        pfx.push_back(cur);
        cur = Perm(n);
        for (int i = 0; i < ps.size(); ++i) {
            sfx.push_back(cur);
            cur = cur * ps[ps.size()-1-i];
        }
        sfx.push_back(cur);
        reverse(sfx.begin(), sfx.end());
        for (int i = 0; i <= m; ++i) {
            for (const auto &g:gen) {
                vector<Perm> qs = ps;
                qs.insert(qs.begin() + i, g);
                int sco = eval(sfx[i] * g * pfx[i]);
                if (best_score < sco) {
                    best_score = sco;
                    nxt = qs;
                    goto exi;
                }
            }
        }
        exi:;
        if (nxt.empty()) break;
        dump(best_score);
        ps = nxt;
        dump(tot(ps).length());
    }
    return tot(ps);
}

int main(int argc, char** argv) {
    // ./a.out X K s[t/f] d[t/f]  dir endless[t/f]
    // [1] X := the number of iterations for creating table
    // [2] K := the width of beam search
    // [3] if s is t, only create table and skip generating result (otherwise set any letter)
    // [4] if d is t, do not touch serialized data
    // [5] file path to save
    // [6] if endless, keep creating table even if all element is filled
    // [7] if true, disalbe fillOrbits
    // [8] if true, disable synthesis
    // [9] if true, do validity check
    // [10] if any, read initial operation
    tablePath = argv[5];
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
    updatePermMoveMap();

    Perm initialP = Perm(n);
    vector<Move> initialMoves;
    if (argc >= 11 && strlen(argv[10]) > 0) {
        dump("prepare initial");
        ifstream ifs(argv[10], ios::in);

        string initialMove;
        ifs >> initialMove;
        for (auto& c: initialMove) {
            if (c == '.') c = ' ';
        }
        stringstream ss;
        ss << initialMove;
        string token;
        initialP = Perm(n);
        while (ss >> token) {
            string s;
            bool inv = token[0] == '-';
            if (inv) s = token.substr(1);
            else s = token;
            Move move{s, inv};
            initialMoves.push_back(move);
            initialP = initialP * m_to_p[move];
        }
    }
    vi updatedInitial = initialP + initialState;
    vector<bool> mismatch(N);
    {
        dump("mismatch");
        int diff = 0;
        for (int i = 0; i < n; ++i) {
            if (updatedInitial[i] != finalState[i]) {
                mismatch[i] = true;
                ++diff;
                dump(i);
            }
        }
        dump(diff);
    }
    Perm reOrder, reOrdInv;
    { // ordering
        reOrder = generateOrdering(puzzleType, N, mismatch);
        reOrdInv = reOrder.inv();
        finalState = reOrder + finalState;
        initialState = reOrder + initialState;

        vector<Perm> genNew;
        for (const auto& permOld: gen) {
            vi p = permOld.p;
            vi q(N);
            for (int j = 0; j < N; ++j) {
                q[reOrdInv.p[j]] = reOrdInv.p[p[j]];
            }
            genNew.push_back(Perm(q, permOld.oper));
        }
        gen = genNew;
        updatePermMoveMap();
        initialP = Perm(n);
        for (auto move: initialMoves) {
            initialP = initialP * m_to_p[move];
        }
        updatedInitial = initialP + initialState;

        bool misStart = false;
        for (int i = 0; i < n; ++i) {
            if (updatedInitial[i] != finalState[i]) {
                misStart = true;
            } else if (misStart){
                assert(false);
            }
        }
    }

    string sampleAns;
    cin>>sampleAns;
    vi sampleTarget = generateSampleTarget(sampleAns, m_to_p, N);

    generatePreFoundCombo(gen, puzzleType);
    bool dry = false;
    if (argc >= 5 && argv[4][0] == 't') {
        dry = true;
    }
    vi base(n);
    for (int i = 0; i < n; ++i) base[i] = i;
    int targetCnt = calcTargetCnt(puzzleType, n);
    dump(targetCnt);
    bool enableOrbits = true, enableSynthesis = true, cubeValidityCheck = false;
    if (argc >= 7 && argv[6][0] == 't') {
        targetCnt = 1e9;
        dump("targetCnt reset");
    }
    if (argc >= 8 && argv[7][0] == 't') {
        enableOrbits = false;
    }
    if (argc >= 9 && argv[8][0] == 't') {
        enableSynthesis= false;
    }
    if (argc >= 10 && argv[9][0] == 't') {
        dump("validity check");
        cubeValidityCheck = true;
    }
    auto table = loadTable(puzzleType, n, false);
    prl;
    createTable(table, gen, base, atoi(argv[1]), 1000, 10, targetCnt, enableOrbits, enableSynthesis, cubeValidityCheck);
    prl;
    if (argc >= 4 && argv[3][0]=='t') {
        return 0;
    }
    if (!dry) serializeTable(table, puzzleType);

    prl;
    auto groups = createGroups(updatedInitial, finalState, table, puzzleType, reOrder);
    prl;
    vector<vi> target(n);
    dump("groups");
    for (auto e:groups){
        auto [B, A] = e;
        debug(A.begin(), A.end());
        debug(B.begin(), B.end());
        for (auto a: A) {
            target[a] = B;
        }
    }
    dump("sampleTarget");
    debug(sampleTarget.begin(), sampleTarget.end());

    auto res = factorize(base, table, target, atoi(argv[2]), sampleTarget, puzzleType[0] == 'c', gen);
    dump(initialP.length());
    res = initialP * res;
    cerr << res << endl;

    dump(res.length());
    if (true) {
        vi curPerm(n), curState = initialState;
        for (int i = 0; i < n; ++i) curPerm[i] = i;
        // for (int i = (int)res.oper.size()-1; i >= 0; --i) {
        //     assert(m_to_p.count(res.oper[i]));
        //     curPerm = m_to_p[res.oper[i]] * curPerm;
        // }
        for (int i = 0; i < res.oper.size(); ++i) {
            assert(m_to_p.count(res.oper[i]));
            curState = m_to_p[res.oper[i]] + curState;
        }
        // debug(curPerm.begin(), curPerm.end());
        // debug(transPerm.begin(), transPerm.end());
        // assert(curPerm == transPerm);

        debug(curState.begin(), curState.end());
        debug(finalState.begin(), finalState.end());
        assert(curState == finalState);
    }
    string ans;
    for (int i =0; i < res.oper.size(); ++i){
        ans += res.oper[i].toString();
        if (i + 1 < res.oper.size()) ans += '.';
    }
    cout << ans <<endl;
    }

    return 0;
}