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

class WithFFETests extends Config((site, _, _) => { case UnitTests => (q: Parameters) => {
  implicit val p = q
  Seq(
    Module(new FFETest(
      FFEParams(
        dataBits = 8,
        weightBits = 8,
        accBits = 8,
        numTaps = 8,
        initWeights = Seq(1, 3, 15, 74, -74, -15, -3, -1),
        numChannels = 4,
        mmioAddr = 0x1000,
      ),
      timeout = 1000
    )),
  )}
})

class FFETestConfig extends Config(
  new WithFFETests ++
  new BaseSubsystemConfig)
