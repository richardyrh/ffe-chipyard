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


class WithFFETests(accBits: Int, power: Boolean) extends Config((site, _, _) => { case UnitTests => (q: Parameters) => {
  implicit val p = q
  Seq(
    Module(new FFETest(
      FFEParams(
        dataBits = 8,
        weightBits = 8,
        accBits = accBits,
        addedMSBs = -2,
        numTaps = 7,
        // initWeights = Seq(-8, 8, -55, 127, -55, 8, -8), // 200m
        initWeights = Seq(-6, 1, -32, 127, -32, 1, -6), // 100m
        numChannels = 4,
        mmioAddr = 0x1000,
      ),
      timeout = 10000,
      power = power,
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
  new WithFFETests(8, false) ++
  new BaseSubsystemConfig)

class FFEPowerConfig extends Config(
  new WithFFETests(8, true) ++
  new BaseSubsystemConfig)

class FFE7bAccPowerConfig extends Config(
  new WithFFETests(7, true) ++
  new BaseSubsystemConfig)
