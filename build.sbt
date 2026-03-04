scalaVersion := "2.13.12"

val chiselVersion = "6.5.0"

libraryDependencies ++= Seq(
  "org.chipsalliance" %% "chisel" % chiselVersion,
  // 核心修复：显式添加缺失的 paranamer 依赖
  "com.thoughtworks.paranamer" % "paranamer" % "2.8",
  "org.scalatest" %% "scalatest" % "3.2.16" % "test",
  "edu.berkeley.cs" %% "chiseltest" % "6.0.0" % "test"
)

// 对于 Chisel 6.x，推荐使用这种方式加载插件
addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full)

// 建议添加以下配置，以确保 IDE 和编译器能更好地处理依赖
scalacOptions ++= Seq(
  "-language:reflectiveCalls",
  "-deprecation",
  "-feature",
  "-Xcheckinit"
)