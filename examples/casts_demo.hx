// saturating casts + IEEE specials
fn main() {
    let f = 300.7;
    print(f as i32);
    print(-1.0e300 as i32);
    print((0.0 / 0.0) as i64);
}
