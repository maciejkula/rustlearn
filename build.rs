// Build dependencies.

// Bring in a dependency on an externally maintained `gcc` package which manages
// invoking the C compiler.
extern crate cc;

fn main() {
    cc::Build::new()
        .cpp(true) // Switch to C++ library compilation.
        .file("dependencies/libsvm/svm.cpp")
        .compile("libsvm.a");
}
