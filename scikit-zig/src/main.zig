const std = @import("std");
const testing = std.testing;

pub const regression = @import("regression.zig");

test "regression test" {
    const I = struct { f64, f64 }; 
    const A = regression.linear.ModelImpl(I); 
    var random = std.rand.DefaultPrng.init(0); 
    var r = random.random();
    var a: A = A.init(&r);
    std.log.warn("a: {any}", .{ a }); 
}

test "regression test 2" {
    const I = struct { f64 }; 
    const A = regression.linear.ModelImpl(I); 
    var random = std.rand.DefaultPrng.init(0); 
    var r = random.random();
    var a: A = A.init(&r);
    std.log.warn("a: {any}", .{ a }); 
    const is = [_]I{ I{ 1.0 }, I{ 2.0 }, I{ 3.0 } }; 
    var ys = [_]f64{ 1.0, 2.0, 3.0 }; 
    const test_allocator = std.testing.allocator; 
    try a.fit(&is, &ys, test_allocator); 
    std.log.warn("a: {any}", .{ a }); 
}