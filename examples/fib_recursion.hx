// recursion + early return + else-if chains (frontend/sema/interpreter exercise)
fn fib(n: i64) -> i64 {
    if n < 2 {
        return n;
    } else if n < 15 {
        return fib(n - 1) + fib(n - 2);
    }
    return fib(n - 3) + 2 * fib(n - 2) - fib(n - 4) + 4;
}

fn main() {
    print(fib(24));
}
