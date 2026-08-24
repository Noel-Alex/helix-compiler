// && || short-circuit evaluation must be observable
fn side(v: i64, hit: [i64]) -> bool {
    hit[v] = hit[v] + 1;
    return v == 3;
}

fn main() {
    let hits: [i64] = zeros(5);
    let r = side(1, hits) && side(3, hits);   // second call runs (first true)
    print(r);
    print(hits[1]);
    print(hits[3]);
    let q = side(2, hits) || side(4, hits);   // first false, second runs
    print(q);
    print(hits[2]);
    print(hits[4]);
}
