package chipyard.unittest

import chisel3._
import org.chipsalliance.cde.config._
import freechips.rocketchip.subsystem.{BaseSubsystemConfig}
import org.chipsalliance.diplomacy.lazymodule.{LazyModule, LazyModuleImp}
import freechips.rocketchip.devices.tilelink._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util._
import org.chipsalliance.cde.config.Parameters
import chisel3.util._
import chiseltest._
import freechips.rocketchip.unittest._
import chipyard.example._
import scala.io.Source


object FFEUtils {
  def readTapWeights(filename: String = "taps.hex", numTaps: Int = 8): Seq[Int] = {
    try {
      val source = Source.fromFile(filename)
      val weights = source.getLines().map(line => Integer.parseInt(line.trim, 16)).take(numTaps).toSeq
      source.close()
      if (weights.length < numTaps) {
        println(s"Warning: Only ${weights.length} weights found in $filename, padding with zeros")
        weights ++ Seq.fill(numTaps - weights.length)(0)
      } else {
        println(s"Read $numTaps weights from $filename")
        weights
      }
    } catch {
      case e: Exception =>
        println(s"Error reading tap weights from $filename: ${e.getMessage}")
        println("Using default weights (all zeros)")
        Seq.fill(numTaps)(0)
    }
  }
}

class WithFFETests extends Config((site, _, _) => { case UnitTests => (q: Parameters) => {
  implicit val p = q
  Seq(
    Module(new FFETest(
      FFEParams(
        dataBits = 8,
        weightBits = 8,
        accBits = 8,
        numTaps = 8,
        initWeights = FFEUtils.readTapWeights("generators/chipyard/src/main/resources/memory/taps.hex"),
        numChannels = 4,
        mmioAddr = 0x1000,
      ),
      timeout = 1000
    )),
  )}
})


/**
  * This config is used to test the FFE module.
  * 
  * Example usage:
  * 
  * ```bash
  * make SUBPROJECT=ffe CONFIG=FFETestConfig BINARY=none LOADMEM=1 run-binary-debug
  * ```
  */
class FFETestConfig extends Config(
  new WithFFETests ++
  new BaseSubsystemConfig)
