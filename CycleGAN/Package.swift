// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "CycleGAN",
    products: [
        // Products define the executables and libraries produced by a package, and make them visible to other packages.
        .library(
            name: "CycleGAN",
            targets: ["CycleGAN"]),
    ],
    dependencies: [
        .package(url: "https://github.com/mxcl/Path.swift", from: "0.16.1"),
        .package(url: "https://github.com/saeta/Just", from: "0.7.2")
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "CycleGAN",
            dependencies: ["Path", "Just"]),
        .testTarget(
            name: "CycleGANTests",
            dependencies: ["CycleGAN"]),
    ]
)
