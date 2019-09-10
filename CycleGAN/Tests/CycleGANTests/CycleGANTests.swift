import XCTest
@testable import CycleGAN

final class CycleGANTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(CycleGAN().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
