# Optpus

Optpus is heuristic zoo for optimization problems.


## どういう風に使いたいか

```rs
let prob = MaxCut::load_from_file("filename");
let mut state = SearchState::new(prob);
let bs = beam_search::<MaxCutFlip>::new("setting");
bs.run(state);
```

trait

* `Problem`: 問題に関する trait, いらない？
* `Solution<Problem>`: 問題の解に関する trait, いらない？
* `Neighbor<Solution<Problem>>`: 隣接解を生成する trait
  * `iter(sol: &Solution<Problem>) -> Iterator<Item=Move>`
* `NeighborApply<Neigbor>`: Solution に追加する trait
  * `apply(&mut self, move: Neighbor)`
* `State<Solution<Problem>>`: 