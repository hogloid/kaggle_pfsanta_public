pub struct UnionFind {
    root: Vec<i32>,
}

impl UnionFind {
    pub fn new(len: usize) -> UnionFind {
        UnionFind { root: vec![-1; len] }
    }

    pub fn root(&self, mut p: usize) -> usize {
        while self.root[p] >= 0 {
            p = self.root[p] as usize;
        }
        p
    }

    pub fn join(&mut self, p: usize, q: usize) -> bool {
        let p = self.root(p);
        let q = self.root(q);
        if p == q {
            return false;
        }
        if self.root[p] < self.root[q] {
            self.root[p] += self.root[q];
            self.root[q] = p as i32;
        } else {
            self.root[q] += self.root[p];
            self.root[p] = q as i32;
        }
        true
    }
}