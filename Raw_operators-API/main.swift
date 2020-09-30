import Foundation
import TensorFlow

// print(_Raw.mul(Tensor([2.0, 3.0]), Tensor([5.0, 6.0])))


print("Defining a new operator (.*)")

infix operator .* : MultiplicationPrecedence

extension Tensor where Scalar: Numeric {
    @differentiable(where Scalar: TensorFlowFloatingPoint)
    // @differentiable specify the derivative function type (In this case "TensorFlowFloatingPoint")
    static func .* (_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
        return _Raw.mul(lhs, rhs)
    }
}

extension Tensor where Scalar : TensorFlowFloatingPoint {
    @derivative(of: .*)
    // @derivative specify the original function (In this case ".*")
    static func multiplyDerivative(
        _ lhs: Tensor, _ rhs: Tensor
    ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        return (lhs * rhs, { v in
            ((rhs * v).unbroadcasted(to: lhs.shape),
            (lhs * v).unbroadcasted(to: rhs.shape))
        })
    }
}

let x: Tensor<Double> = [[1.0, 2.0], [3.0, 4.0]]
let y: Tensor<Double> = [[8.0, 7.0], [6.0, 5.0]]

print("x is: ",x)
print("y is: ",y)
print("and x .* y is: ",x .* y)

var answer = gradient(at: x, y) { x, y in
    (x .* y).sum()
}
print("(x .* y).sum() is: ",answer)
