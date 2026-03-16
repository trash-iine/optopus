use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::search_state::{ProblemTrait, Rankable};

/// 充足された節数を最大化する解（変数は 0-indexed、節のリテラルは 1-indexed の符号付き整数）
#[derive(Debug, Clone)]
pub struct SatSolution {
    /// 変数の割り当て (x[i] = 変数 i+1 の真偽値)
    pub x: Vec<bool>,
    /// gain[i] = x[i] をフリップしたときの n_satisfied の変化量 (正 = 改善)
    pub gain: Vec<i64>,
    /// 現在充足されている節の数
    pub n_satisfied: usize,
}

impl Rankable for SatSolution {
    // MaxSAT: 充足節数が多い方が優れた解
    fn is_better_than(&self, other: &Self) -> bool {
        self.n_satisfied > other.n_satisfied
    }
}

/// DIMACS CNF 形式の MaxSAT 問題
#[derive(Debug, Clone)]
pub struct Sat {
    n_vars: usize,
    /// 全節 (リテラルは符号付き整数、変数は 1-indexed)
    clauses: Vec<Vec<i64>>,
    /// clauses_per_var[i] = 変数 i+1 を含む節のインデックス一覧 (0-indexed)
    clauses_per_var: Vec<Vec<usize>>,
}

impl Sat {
    pub fn new(n_vars: usize) -> Self {
        Self {
            n_vars,
            clauses: Vec::new(),
            clauses_per_var: vec![vec![]; n_vars],
        }
    }

    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    pub fn n_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// 節を追加する。リテラルは符号付き整数 (1-indexed 変数)
    pub fn add_clause(&mut self, literals: impl IntoIterator<Item = i64>) {
        let clause_idx = self.clauses.len();
        let clause: Vec<i64> = literals.into_iter().collect();

        for &lit in &clause {
            let var_idx = lit.unsigned_abs() as usize - 1;
            if var_idx < self.n_vars {
                self.clauses_per_var[var_idx].push(clause_idx);
            }
        }
        self.clauses.push(clause);
    }

    /// 節 idx のリテラル列を返す
    pub fn clause(&self, idx: usize) -> &[i64] {
        &self.clauses[idx]
    }

    /// 全節のイテレータ
    pub fn all_clauses(&self) -> impl Iterator<Item = &[i64]> {
        self.clauses.iter().map(|c| c.as_slice())
    }

    /// 変数 i (0-indexed) を含む節の節インデックス一覧
    pub fn clause_indices_of_var(&self, i: usize) -> &[usize] {
        &self.clauses_per_var[i]
    }

    /// 変数 i (0-indexed) を含む節のリテラル列のイテレータ
    pub fn clauses_of_var(&self, i: usize) -> impl Iterator<Item = &[i64]> {
        self.clauses_per_var[i]
            .iter()
            .map(|&idx| self.clauses[idx].as_slice())
    }

    /// 節が充足されているか判定する
    pub fn is_clause_satisfied(clause: &[i64], x: &[bool]) -> bool {
        clause.iter().any(|&lit| {
            let var = lit.unsigned_abs() as usize - 1;
            x[var] == (lit > 0)
        })
    }

    /// 充足節数を計算する
    pub fn calc_satisfied(&self, x: &[bool]) -> usize {
        self.clauses
            .iter()
            .filter(|c| Self::is_clause_satisfied(c, x))
            .count()
    }

    /// 変数 i (0-indexed) をフリップしたときの充足節数の変化量を計算する
    pub fn calc_gain(&self, x: &[bool], i: usize) -> i64 {
        let mut gain = 0i64;
        for &clause_idx in &self.clauses_per_var[i] {
            let clause = &self.clauses[clause_idx];

            // この節における変数 i のリテラルと現在の充足状態
            let i_lit = *clause
                .iter()
                .find(|&&lit| lit.unsigned_abs() as usize - 1 == i)
                .expect("clauses_per_var is inconsistent");
            let i_lit_sat = x[i] == (i_lit > 0);

            // i 以外のリテラルが充足されているか
            let other_sat = clause.iter().any(|&lit| {
                let var = lit.unsigned_abs() as usize - 1;
                if var == i {
                    return false;
                }
                x[var] == (lit > 0)
            });

            let was_sat = i_lit_sat || other_sat;
            let will_be_sat = !i_lit_sat || other_sat; // i をフリップした後

            gain += will_be_sat as i64 - was_sat as i64;
        }
        gain
    }

    /// 変数 flipped (0-indexed) がフリップされた状態で変数 j の gain を計算する
    /// (解を変更せずに計算するため、x は元のまま使用する)
    pub fn calc_gain_with_virtual_flip(&self, x: &[bool], flipped: usize, j: usize) -> i64 {
        let mut gain = 0i64;
        for &clause_idx in &self.clauses_per_var[j] {
            let clause = &self.clauses[clause_idx];

            let j_lit = *clause
                .iter()
                .find(|&&lit| lit.unsigned_abs() as usize - 1 == j)
                .expect("clauses_per_var is inconsistent");
            let j_lit_sat = x[j] == (j_lit > 0);

            // flipped 変数は仮想的にフリップした値を使う
            let other_sat = clause.iter().any(|&lit| {
                let var = lit.unsigned_abs() as usize - 1;
                if var == j {
                    return false;
                }
                let x_var = if var == flipped { !x[var] } else { x[var] };
                x_var == (lit > 0)
            });

            let was_sat = j_lit_sat || other_sat;
            let will_be_sat = !j_lit_sat || other_sat;

            gain += will_be_sat as i64 - was_sat as i64;
        }
        gain
    }

    /// DIMACS CNF 形式のファイルを読み込む
    pub fn load_file(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        let mut n_clauses = 0usize;
        let mut sat = None::<Self>;

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('c') {
                continue; // コメント行をスキップ
            }
            if trimmed.starts_with('p') {
                // "p cnf <n_vars> <n_clauses>"
                let mut iter = trimmed.split_whitespace();
                iter.next(); // "p"
                iter.next(); // "cnf"
                let n_vars: usize = iter.next().ok_or("Not found n_vars")?.parse()?;
                n_clauses = iter.next().ok_or("Not found n_clauses")?.parse()?;
                sat = Some(Self::new(n_vars));
                continue;
            }
            let s = sat.as_mut().ok_or("Header line not found before clauses")?;
            // 節行: スペース区切りのリテラル列、0 で終端
            let literals: Vec<i64> = trimmed
                .split_whitespace()
                .map(|t| t.parse::<i64>())
                .collect::<Result<_, _>>()?;
            let clause: Vec<i64> = literals.into_iter().take_while(|&v| v != 0).collect();
            if !clause.is_empty() {
                s.add_clause(clause);
            }
        }

        let s = sat.ok_or("Empty file")?;
        if s.n_clauses() != n_clauses {
            return Err(format!(
                "Expected {} clauses, got {}",
                n_clauses,
                s.n_clauses()
            )
            .into());
        }
        Ok(s)
    }
}

impl ProblemTrait for Sat {
    type Solution = SatSolution;

    fn new_solution(&self, rng: &mut impl rand::Rng) -> SatSolution {
        let x: Vec<bool> = (0..self.n_vars).map(|_| rng.random_bool(0.5)).collect();
        let gain: Vec<i64> = (0..self.n_vars).map(|i| self.calc_gain(&x, i)).collect();
        let n_satisfied = self.calc_satisfied(&x);
        SatSolution { x, gain, n_satisfied }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3変数・3節のSATインスタンスを作成する
    /// (x1 ∨ x2), (¬x1 ∨ x3), (¬x2 ∨ ¬x3)
    fn make_sat() -> Sat {
        let mut sat = Sat::new(3);
        sat.add_clause([1, 2]);       // x1 ∨ x2
        sat.add_clause([-1, 3]);      // ¬x1 ∨ x3
        sat.add_clause([-2, -3]);     // ¬x2 ∨ ¬x3
        sat
    }

    #[test]
    fn test_calc_satisfied() {
        let sat = make_sat();
        // x = [true, false, true]: (T∨F)=T, (F∨T)=T, (T∨F)=T → 全充足
        assert_eq!(sat.calc_satisfied(&[true, false, true]), 3);
        // x = [false, false, false]: (F∨F)=F, (T∨F)=T, (T∨T)=T → 2充足
        assert_eq!(sat.calc_satisfied(&[false, false, false]), 2);
    }

    #[test]
    fn test_calc_gain_matches_delta() {
        let sat = make_sat();
        let x = vec![true, false, true];
        let n_sat = sat.calc_satisfied(&x);
        for i in 0..3 {
            let mut x2 = x.clone();
            x2[i] = !x2[i];
            let expected_delta = sat.calc_satisfied(&x2) as i64 - n_sat as i64;
            assert_eq!(
                sat.calc_gain(&x, i),
                expected_delta,
                "gain[{}] mismatch",
                i
            );
        }
    }

    #[test]
    fn test_n_clauses() {
        let sat = make_sat();
        assert_eq!(sat.n_clauses(), 3);
        assert_eq!(sat.n_vars(), 3);
    }
}
