use std::ops::{Mul, Not};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Permutation {
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
}

impl Permutation {
    pub fn new(perm: Vec<usize>) -> Permutation {
        let mut perm_inv = vec![0; perm.len()];
        for i in 0..perm.len() {
            perm_inv[perm[i]] = i;
        }
        Permutation { perm, perm_inv }
    }

    pub fn identity(size: usize) -> Permutation {
        Permutation {
            perm: (0..size).collect(),
            perm_inv: (0..size).collect(),
        }
    }

    pub fn sparsify(&self) -> SparsePermutation {
        let mut src = vec![];
        let mut dest = vec![];
        for i in 0..self.len() {
            if self.perm[i] != i {
                src.push(i);
                dest.push(self.perm[i]);
            }
        }
        SparsePermutation { src, dest, len: self.len() }
    }

    pub fn apply<T: Clone>(&self, target: &[T]) -> Vec<T> {
        assert_eq!(self.len(), target.len());
        let mut ret = vec![];
        for i in 0..self.perm.len() {
            ret.push(target[self.perm[i]].clone());
        }
        ret
    }

    pub fn apply_inv<T: Clone>(&self, target: &[T]) -> Vec<T> {
        assert_eq!(self.len(), target.len());
        let mut ret = vec![];
        for i in 0..self.perm.len() {
            ret.push(target[self.perm_inv[i]].clone());
        }
        ret
    }

    pub fn restrict_on(&self, size_after_restriction: usize, restrict_map: &[Option<usize>]) -> Permutation {
        let mut ret = vec![0; size_after_restriction];
        for i in 0..self.len() {
            match (restrict_map[i], restrict_map[self.perm[i]]) {
                (Some(p), Some(q)) => ret[p] = q,
                (None, None) => (),
                _ => panic!(),
            }
        }
        Permutation::new(ret)
    }

    // check if this permutation moves elements only in the restriction set
    pub fn is_restriction_faithful(&self, _size_after_restriction: usize, restrict_map: &[Option<usize>]) -> bool {
        for i in 0..self.len() {
            if i == self.perm[i] {
                continue;
            }
            if restrict_map[i].is_none() || restrict_map[self.perm[i]].is_none() {
                return false;
            }
        }
        true
    }

    pub fn perm(&self, i: usize) -> usize {
        self.perm[i]
    }

    pub fn perm_inv(&self, i: usize) -> usize {
        self.perm_inv[i]
    }

    pub unsafe fn perm_inv_unchecked(&self, i: usize) -> usize {
        *self.perm_inv.get_unchecked(i)
    }

    pub fn len(&self) -> usize {
        self.perm.len()
    }
}

impl Not for Permutation {
    type Output = Permutation;

    fn not(self) -> Self::Output {
        Permutation {
            perm: self.perm_inv,
            perm_inv: self.perm,
        }
    }
}

// Composition of permutations.
// (x * y).apply(seq) == x.apply(y.apply(seq)) holds.
impl Mul<&Permutation> for &Permutation {
    type Output = Permutation;

    fn mul(self, rhs: &Permutation) -> Self::Output {
        // a.apply(b.apply(target))[i] == b.apply(target)[a.perm[i]] == target[b.perm[a.perm[i]]]
        // (a * b).apply[i] == target[(a * b).perm[i]]
        assert_eq!(self.len(), rhs.len());
        let mut perm = vec![];
        for i in 0..self.len() {
            perm.push(rhs.perm[self.perm[i]]);
        }
        Permutation::new(perm)
    }
}

impl Mul<Permutation> for &Permutation {
    type Output = Permutation;

    fn mul(self, rhs: Permutation) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&Permutation> for Permutation {
    type Output = Permutation;

    fn mul(self, rhs: &Permutation) -> Self::Output {
        &self * rhs
    }
}

impl Mul<Permutation> for Permutation {
    type Output = Permutation;

    fn mul(self, rhs: Permutation) -> Self::Output {
        &self * &rhs
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SparsePermutation {
    src: Vec<usize>,
    dest: Vec<usize>,
    len: usize,
}

impl SparsePermutation {
    pub fn new(src: Vec<usize>, dest: Vec<usize>, len: usize) -> SparsePermutation {
        assert_eq!(src.len(), dest.len());
        SparsePermutation { src, dest, len }
    }

    pub fn densify(&self) -> Permutation {
        let mut perm = (0..self.len).collect::<Vec<_>>();
        for i in 0..self.src.len() {
            perm[self.src[i]] = self.dest[i];
        }
        Permutation::new(perm)
    }

    pub fn apply<T: Clone>(&self, target: &[T]) -> Vec<T> {
        assert_eq!(self.len(), target.len());
        let mut ret = target.to_owned();
        for i in 0..self.src.len() {
            ret[self.src[i]] = target[self.dest[i]].clone();
        }
        ret
    }

    pub fn apply_inv<T: Clone>(&self, target: &[T]) -> Vec<T> {
        assert_eq!(self.len(), target.len());
        let mut ret = target.to_owned();
        for i in 0..self.src.len() {
            ret[self.dest[i]] = target[self.src[i]].clone();
        }
        ret
    }

    pub fn apply_inplace<T: Clone>(&self, target: &mut [T]) {
        assert_eq!(self.len(), target.len());
        let mut buf = vec![];
        for i in 0..self.src.len() {
            buf.push(target[self.dest[i]].clone());
        }
        for (i, val) in buf.into_iter().enumerate() {
            target[self.src[i]] = val;
        }
    }

    pub fn apply_inv_inplace<T: Clone>(&self, target: &mut [T]) {
        assert_eq!(self.len(), target.len());
        let mut buf = vec![];
        for i in 0..self.src.len() {
            buf.push(target[self.src[i]].clone());
        }
        for (i, val) in buf.into_iter().enumerate() {
            target[self.dest[i]] = val;
        }
    }

    // Compute permutation b s.t. b * g == g * self.
    // b = g * self * g^-1
    pub fn conjugate(&self, g: &Permutation) -> SparsePermutation {
        assert_eq!(self.len(), g.len());

        let mut src = vec![];
        let mut dest = vec![];
        for i in 0..self.src.len() {
            src.push(g.perm_inv[self.src[i]]);
            dest.push(g.perm_inv[self.dest[i]]);
        }

        SparsePermutation { src, dest, len: self.len() }
    }

    pub fn restrict_on(&self, size_after_restriction: usize, restrict_map: &[Option<usize>]) -> SparsePermutation {
        let mut src = vec![];
        let mut dest = vec![];

        for i in 0..self.src.len() {
            match (restrict_map[self.src[i]], restrict_map[self.dest[i]]) {
                (Some(p), Some(q)) => {
                    src.push(p);
                    dest.push(q);
                }
                (None, None) => (),
                _ => panic!(),
            }
        }

        SparsePermutation { src, dest, len: size_after_restriction }
    }

    // check if this permutation moves elements only in the restriction set
    pub fn is_restriction_faithful(&self, _size_after_restriction: usize, restrict_map: &[Option<usize>]) -> bool {
        for i in 0..self.src.len() {
            assert_ne!(self.src[i], self.dest[i]);
            if restrict_map[self.src[i]].is_none() || restrict_map[self.dest[i]].is_none() {
                return false;
            }
        }
        true
    }

    pub fn normalize(&self) -> SparsePermutation {
        let mut sd = vec![];
        for i in 0..self.src.len() {
            sd.push((self.src[i], self.dest[i]));
        }
        sd.sort();
        let mut src = vec![];
        let mut dest = vec![];
        for (s, d) in sd {
            src.push(s);
            dest.push(d);
        }

        SparsePermutation { src, dest, len: self.len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn num_moves(&self) -> usize {
        self.src.len()
    }

    pub fn src(&self, i: usize) -> usize {
        self.src[i]
    }

    pub fn dest(&self, i: usize) -> usize {
        self.dest[i]
    }

    pub unsafe fn src_unchecked(&self, i: usize) -> usize {
        *self.src.get_unchecked(i)
    }

    pub unsafe fn dest_unchecked(&self, i: usize) -> usize {
        *self.dest.get_unchecked(i)
    }
}

impl Not for SparsePermutation {
    type Output = SparsePermutation;

    fn not(self) -> Self::Output {
        SparsePermutation {
            src: self.dest,
            dest: self.src,
            len: self.len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation() {
        let a = Permutation::new(vec![1, 2, 3, 0]);
        let test_seq = &["a", "b", "c", "d"];
        assert_eq!(a.len(), 4);
        assert_eq!(a.apply(test_seq), vec!["b", "c", "d", "a"]);
        assert_eq!(a.apply_inv(test_seq), vec!["d", "a", "b", "c"]);
        assert_eq!((!a.clone()).apply(test_seq), vec!["d", "a", "b", "c"]);

        let b = Permutation::new(vec![2, 0, 3, 1]);
        assert_eq!(a.apply(&b.apply(test_seq)), (&a * &b).apply(test_seq));
        assert_eq!(b.apply(&a.apply(test_seq)), (&b * &a).apply(test_seq));

        assert_eq!(!(&a * &b), !b.clone() * !a.clone());
    }

    #[test]
    fn test_sparse_permutation() {
        let a = Permutation::new(vec![0, 6, 2, 1, 4, 5, 3]);
        let a_sparse = a.sparsify();

        assert_eq!(a_sparse.densify(), a);

        let test_seq = vec!["a", "b", "c", "d", "e", "f", "g"];
        assert_eq!(a.apply(&test_seq), a_sparse.apply(&test_seq));

        let mut tmp = test_seq.clone();
        a_sparse.apply_inplace(&mut tmp);
        assert_eq!(a_sparse.apply(&test_seq), tmp);
        (!a_sparse.clone()).apply_inplace(&mut tmp);
        assert_eq!(test_seq, tmp);

        assert_eq!(a_sparse.apply_inv(&test_seq), (!a_sparse.clone()).apply(&test_seq));
        a_sparse.apply_inv_inplace(&mut tmp);
        assert_eq!(a_sparse.apply_inv(&test_seq), tmp);
    }

    #[test]
    fn test_conjugate() {
        let perms = vec![
            Permutation::new(vec![0, 6, 2, 1, 4, 5, 3]),
            Permutation::new(vec![1, 3, 5, 2, 4, 6, 0]),
            Permutation::new(vec![2, 1, 3, 4, 6, 0, 5]),
            Permutation::new(vec![3, 1, 4, 5, 6, 0, 2]),
        ];

        for i in 0..perms.len() {
            for j in 0..perms.len() {
                let b = perms[i].sparsify().conjugate(&perms[j]);
                assert_eq!(b.densify(), &perms[j] * &perms[i] * !perms[j].clone());
                assert_eq!(b.densify() * &perms[j], &perms[j] * &perms[i]);
            }
        }
    }
}
