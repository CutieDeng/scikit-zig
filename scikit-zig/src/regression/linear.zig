const std = @import("std");

pub fn ModelImpl(comptime XType: type) type {
    const x_type_info = @typeInfo(XType);
    switch (x_type_info) {
        .Struct => {},
        else => @compileError("XType must be a struct"),
    }
    const local_x_fields = std.meta.fields(XType);
    inline for (local_x_fields) |local_field| {
        if (local_field.type != f64) {
            @compileError("XType must be a struct of f64");
        }
    }
    return struct {
        pub const X = XType;
        pub const x_fields = local_x_fields;
        inner: XType,
        const Self = @This();
        pub fn init(rand: *std.rand.Random) Self {
            var inner: XType = undefined;
            inline for (x_fields) |field| {
                @field(inner, field.name) = rand.float(f64);
            }
            return Self{ .inner = inner };
        }
        pub fn fit(self: *Self, x: []const X, y: []const f64, gpa: std.mem.Allocator) !void {
            @setRuntimeSafety(true);
            if (x.len != y.len) unreachable;
            const buffer = try gpa.alloc(f64, x.len);
            defer gpa.free(buffer);
            const minimum_batch = 100;
            const gradient = 0.0001;
            var last_error_val: ?f64 = null;
            while (true) {
                const error_val = self.square_error(x, y);
                if (last_error_val) |l| {
                    if (l - error_val <= 1e-6)
                        break;
                }
                last_error_val = error_val;
                for (0..minimum_batch) |_|
                    self.iter(x, y, gradient, buffer);
            }
        }
        pub fn predict(self: Self, x: []const X, allocator: std.mem.Allocator) ![]f64 {
            const result = try allocator.alloc(f64, x.len);
            errdefer allocator.free(result);
            for (x, result) |xi, *ri| {
                ri.* = Self.multi(self.inner, xi);
            }
            return result;
        }
        fn iter(self: *Self, x: []const X, y: []const f64, gradient: f64, buffer: []f64) void {
            if (x.len != y.len) unreachable;
            if (x.len != buffer.len) unreachable;
            const len: f64 = @floatFromInt(x.len);
            for (x, y, buffer) |xi, yi, *bi| {
                const multi_val = Self.multi(self.inner, xi);
                const diff = multi_val - yi;
                bi.* = diff / len;
            }
            inline for (x_fields) |field| {
                const field_name = field.name;
                var sum: f64 = 0.0;
                for (x, buffer) |xi, bi| {
                    const xi_value: f64 = @field(xi, field_name);
                    sum += xi_value * bi;
                }
                const step = gradient * sum;
                const field_value: f64 = @field(self.inner, field_name);
                @field(self.inner, field_name) = field_value - step;
            }
        }
        fn square_error(self: *Self, x: []const X, y: []const f64) f64 {
            if (x.len != y.len) unreachable;
            var result: f64 = 0.0;
            const a1: f64 = @floatFromInt(x.len);
            const average = a1 * a1;
            for (x, y) |xi, yi| {
                const multi_val = Self.multi(self.inner, xi);
                const diff = multi_val - yi;
                const d2 = diff * diff;
                const d3 = d2 / average;
                result += d3;
            }
            return result;
        }
        fn multi(lhs: X, rhs: X) f64 {
            var result: f64 = 0.0;
            inline for (x_fields) |field| {
                const field_name = field.name;
                var lhs_value: f64 = @field(lhs, field_name);
                var rhs_value: f64 = @field(rhs, field_name);
                result += lhs_value * rhs_value;
            }
            return result;
        }
    };
}
