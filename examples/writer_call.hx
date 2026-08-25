// A loop whose body calls a user function with side effects (a print).
// The analysis side-effect gate only recognized `print` inline, so this
// used to get an unsound SAFE verdict. User calls must veto: the backend
// demotes regions containing them, but the VERDICT must say so.
fn tag(x: i64) -> i64 {
    print(x);
    return x;
}
fn main() {
    let a: [i64] = zeros(8);
    for i in 0..8 {
        a[i] = tag(i);   // side effect inside the hot loop
    }
    print(a[3]);
}
