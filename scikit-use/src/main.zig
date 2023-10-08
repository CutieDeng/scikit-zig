const std = @import("std");

const scikit = @import("scikit-lib");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const F64x3 = struct { f64, f64, f64 };
    const Model = scikit.regression.linear.ModelImpl(F64x3);

    var rng = std.rand.DefaultPrng.init(0);
    var rand = rng.random();

    var model = Model.init(&rand);
    // determine the function like 5 * x0 + 2 * x1 - 3 * x2 = y ~
    // x0, x1, x2 ranges from 0 ~ 100
    var X: std.ArrayList(F64x3) = undefined;
    X = std.ArrayList(F64x3).init(allocator);
    defer X.deinit();
    {
        const length = 1000;
        for (0..length) |_| {
            var x = .{
                rand.float(f64) * 100.0,
                rand.float(f64) * 100.0,
                rand.float(f64) * 100.0,
            };
            try X.append(x);
        }
    }
    var y: std.ArrayList(f64) = undefined;
    y = std.ArrayList(f64).init(allocator);
    defer y.deinit();
    {
        for (X.items) |x| {
            var y_ = 5.0 * x.@"0" + 2.0 * x.@"1" - 3.0 * x.@"2";
            y_ += rand.floatNorm(f64) * 0.1;
            try y.append(y_);
        }
    }

    try model.fit(X.items, y.items, allocator);

    std.debug.print("model: {any}\n", .{model});
}
