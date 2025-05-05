package chipyard.example
import chisel3.experimental.{annotate, ChiselAnnotation}
import firrtl.annotations._
import firrtl.transforms._
import firrtl.RenameMap

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.{Config, Field, Parameters}
import org.chipsalliance.diplomacy.lazymodule._
import freechips.rocketchip.diplomacy.{AddressSet, IdRange}
import freechips.rocketchip.regmapper.{HasRegMap, RegField}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.unittest._
import freechips.rocketchip.resources.{SimpleDevice, BigIntHexContext}
import scala.io.Source


/**
  * Parameters for the Ethernet FIR filter
  *
  * @param dataBits - Number of bits in the input data
  * @param weightBits - Number of bits in the weights
  * @param accBits - Number of bits in the accumulator
  * @param addedMSBs - Number of MSBs in the accumulator that used to accumulation overflow
  * @param numTaps - Number of taps in the filter
  * @param initWeights - Initial weights for the filter
  * @param numChannels - Number of channels (taps) in the filter
  * @param mmioAddr - MMIO address for the filter
  */
case class FFEParams(
  dataBits: Int = 8,
  weightBits: Int = 8,
  accBits: Int = 8,
  addedMSBs: Int = 2,
  numTaps: Int = 8,
  initWeights: Seq[Int],
  numChannels: Int = 4,
  mmioAddr: BigInt = 0,
)

/**
  * FIR filter
  *
  * Functionally this is equivalent with the following circuit:
  *
  * val y_0 = io.weights(7) * io.in
  * val y_1 = RegNext(y_0 + io.weights(6) * io.in)
  * val y_2 = RegNext(y_1 + io.weights(5) * io.in)
  * val y_3 = RegNext(y_2 + io.weights(4) * io.in)
  * val y_4 = RegNext(y_3 + io.weights(3) * io.in)
  * val y_5 = RegNext(y_4 + io.weights(2) * io.in)
  * val y_6 = RegNext(y_5 + io.weights(1) * io.in)
  * val y_7 = RegNext(y_6 + io.weights(0) * io.in)
  * io.out := y_7
  *
  * @param params: the configuration of the FIR filter
  */
class FirFilter(params: FFEParams) extends Module {
  def truncateProduct(x: SInt): SInt = {
    val maxBits = params.weightBits + params.dataBits
    val prodBits = params.accBits - params.addedMSBs
    (x.asTypeOf(UInt(maxBits.W)) >> (maxBits - prodBits)).asTypeOf(SInt(prodBits.W))
  }

  val io = IO(new Bundle {
    val in = Input(SInt(params.dataBits.W))
    val weights = Input(Vec(params.numTaps, SInt(params.weightBits.W)))
    val out = Output(SInt(params.accBits.W))
  })

  val inStage1 = RegNext(io.in)

  val ys = WireInit(0.U.asTypeOf(Vec(params.numTaps, SInt(params.accBits.W))))

  ys(0) := truncateProduct(io.weights(params.numTaps - 1) * inStage1).asTypeOf(ys(0).cloneType)

  for (i <- 1 until params.numTaps) {
    ys(i) := RegNext(ys(i - 1)) + truncateProduct(io.weights(params.numTaps - i - 1) * inStage1)
  }

  io.out := RegNext(ys.last)
}


class FFE(params: FFEParams)(implicit p: Parameters) extends LazyModule {
  val device = new SimpleDevice("ffe", Nil)
  val regNode = TLRegisterNode(Seq(AddressSet(params.mmioAddr, 0xff)), device, beatBytes=8, concurrency=0)

  override lazy val module = new FFEImpl(this)

  class FFEImpl(outer: FFE) extends LazyModuleImp(outer) {
    val io = IO(new Bundle {
      val in = Flipped(ValidIO(Vec(params.numChannels, SInt(params.dataBits.W))))
      val out = ValidIO(Vec(params.numChannels, SInt(params.accBits.W)))
    })

    dontTouch(io)

    val weightRegs = params.initWeights.map(x => RegInit(x.S(params.weightBits.W).asUInt))
    val weightRegFns = weightRegs.zipWithIndex.map { case (reg, idx) =>
      (valid: Bool, bits: UInt) => {
        when (valid) {
          printf("ffe tap %d set to %d\n", idx.U, bits)
          reg := bits.asTypeOf(reg.cloneType)
        }
        true.B // ready
      }
    }
    // assume weights are leq than 8 bits
    outer.regNode.regmap(
      0 -> weightRegFns.map(RegField.w(8, _))
    )

    weightRegs.foreach(dontTouch(_))

    val _firFilters = Seq.tabulate(params.numChannels) { ch =>
      val firFilter = Module(new FirFilter(params))
      firFilter.io.in := io.in.bits(ch)
      firFilter.io.weights := VecInit(weightRegs).asTypeOf(firFilter.io.weights.cloneType)
      io.out.bits(ch) := firFilter.io.out
      dontTouch(firFilter.io)

      firFilter
    }
    // TODO: check number of cycles?
    io.out.valid := io.in.valid && (0 until params.numTaps).foldLeft(io.in.valid)((x, _) => RegNext(x))
  }
}

class FFETestTop(params: FFEParams, timeout: Int)(implicit p: Parameters) extends LazyModule {
  // define client
  val node = TLClientNode(Seq(
    TLMasterPortParameters.v2(
      Seq(TLMasterParameters.v2(
        name = "FFEMMIODriver",
        sourceId = IdRange(0, 8)
      ))
    ),
  ))

  val ffe = LazyModule(new FFE(params))

  ffe.regNode := TLIdentityNode() := node

  lazy val module = new LazyModuleImp(this) with UnitTestModule {
    val ffe_inputs = VecInit({
      val fileSource = scala.io.Source.fromFile("generators/chipyard/src/main/resources/memory/ffe_in.hex")
      val lines = try fileSource.getLines().toArray finally fileSource.close()
      lines.map(line => {
        // Parse each line as hex values for each channel
        val values = line.trim.split("\\s+").map(hex => Integer.parseInt(hex, 16).S(params.dataBits.W))
        VecInit(values.take(params.numChannels))
      })
    })

    val ffe_golden = VecInit({
      val fileSource = scala.io.Source.fromFile("generators/chipyard/src/main/resources/memory/ffe_out.hex")
      val lines = try fileSource.getLines().toArray finally fileSource.close()
      lines.map(line => {
        // Parse each line as hex values for each channel
        val values = line.trim.split("\\s+").map(hex => Integer.parseInt(hex, 16).S(params.accBits.W))
        VecInit(values.take(params.numChannels))
      })
    })

    val ffe_weights = VecInit({
      val fileSource = scala.io.Source.fromFile("generators/chipyard/src/main/resources/memory/taps.hex")
      val lines = try fileSource.getLines().toArray finally fileSource.close()
      lines.map(line => {
        Integer.parseInt(line.trim, 16).S(params.weightBits.W)
      })
    })


    val (sim_counter, _) = Counter(true.B, 10000)
    val (test_sequence_counter, _) = Counter(true.B, ffe_inputs.length)


    ffe.module.io.in.bits := ffe_inputs(test_sequence_counter)
    ffe.module.io.in.valid := true.B

    val (n, e) = node.out.head

    val (legal, a) = e.Put(
      fromSource = 0.U,
      toAddress = (params.mmioAddr + 0).U,
      lgSize = 3.U,
      data = VecInit(ffe_weights.map(_.asUInt + 1.U)).asUInt
    )
    assert(legal)

    n.a.bits := a
    n.a.valid := (sim_counter === 20.U)
    n.d.ready := true.B

    dontTouch(n.a)
    dontTouch(n.d)


    (0 until params.numChannels).foreach(ch => {
      val dutBits = ffe.module.io.out.bits(ch)
      val goldenBits = ffe_golden(test_sequence_counter)(ch)
      when (sim_counter === test_sequence_counter) {
        assert(dutBits === goldenBits.asTypeOf(dutBits.cloneType),
              "At step %d, channel %d: Expected %x, got %x",
              sim_counter.asUInt, ch.asUInt, goldenBits, dutBits)
      }
    })

    val (timeout_counter, _) = Counter(true.B, timeout + 1)
    io.finished := (timeout_counter === timeout.U)
  }
}

class FFETest(params: FFEParams, timeout: Int)(implicit p: Parameters) extends UnitTest(timeout) {
  val dut = Module(LazyModule(new FFETestTop(params, timeout)).module)
  dut.io.start := io.start
  io.finished := dut.io.finished
}
