package chipyard.example

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.{Config, Field, Parameters}
import org.chipsalliance.diplomacy.lazymodule._
import freechips.rocketchip.diplomacy.{AddressSet, IdRange}
import freechips.rocketchip.regmapper.{HasRegMap, RegField}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.unittest._
import freechips.rocketchip.resources.{SimpleDevice, BigIntHexContext}

/**
  * Parameters for the Ethernet FIR filter
  *
  * @param dataBits - Number of bits in the input data
  * @param accBits - Number of bits in the accumulator
  * @param numChannels - Number of channels (taps) in the filter
  * @param mmioAddr - MMIO address for the filter
  */
case class FFEParams(
  dataBits: Int = 8,
  weightBits: Int = 8,
  accBits: Int = 18,
  numTaps: Int = 8,
  initWeights: Seq[Int] = Seq.fill(8)(0),
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
  * @param params
  */
class FirFilter(params: FFEParams) extends Module {
  val io = IO(new Bundle {
    val in = Input(SInt(params.dataBits.W))
    val weights = Input(Vec(params.numTaps, SInt(params.weightBits.W)))
    val out = Output(SInt(params.accBits.W))
  })

  val ys = Wire(Vec(params.numTaps, SInt(params.accBits.W)))

  ys(0) := io.weights(params.numTaps - 1) * io.in
  
  for (i <- 1 until params.numTaps) {
    ys(i) := RegNext(ys(i - 1)) + io.weights(params.numTaps - i - 1) * io.in
  }
  
  io.out := RegNext(ys(params.numTaps - 1))
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
    val (stimCounter, _) = Counter(true.B, 8)
    val (srcCounter, _) = Counter(stimCounter === 0.U, 8)
    val ffeIn = ffe.module.io.in

    // Read hex file into a Scala array
    val ffeInputs = {
      val fileSource = scala.io.Source.fromFile("generators/chipyard/src/main/resources/memory/ffe_in.hex")
      val lines = try fileSource.getLines().toArray finally fileSource.close()
      lines.map(line => {
        // Parse each line as hex values for each channel
        val values = line.trim.split("\\s+").map(hex => Integer.parseInt(hex, 16).S(params.dataBits.W))
        VecInit(values.take(params.numChannels))
      })
    }
    val ffeGolden = {
      val fileSource = scala.io.Source.fromFile("generators/chipyard/src/main/resources/memory/ffe_out.hex")
      val lines = try fileSource.getLines().toArray finally fileSource.close()
      lines.map(line => {
        // Parse each line as hex values for each channel
        val values = line.trim.split("\\s+").map(hex => Integer.parseInt(hex, 16).S(params.accBits.W))
        VecInit(values.take(params.numChannels))
      })
    }
    
    // Convert to hardware lookup table and read using stimCounter
    val ffeInputsVec = VecInit(ffeInputs)
    ffeIn.bits := ffeInputsVec(stimCounter)
    ffeIn.valid := true.B

    val (n, e) = node.out.head
    n.a.valid := false.B
    n.a.bits := 0.U.asTypeOf(n.a.bits.cloneType)
    n.d.ready := true.B
    
    when(stimCounter === 0.U) { // every 8 cycles, write weights
      val (legal, a) = e.Put(
        fromSource = srcCounter,
        toAddress = (params.mmioAddr + 0).U,
        lgSize = 3.U,
        data = x"8badf00ddeadbeef".U
      )
      assert(legal)
      n.a.bits := a
      n.a.valid := true.B
    }
    dontTouch(n.a)

    // check if output matches our golden result
    for (t <- 0 until 10) {
      when(stimCounter === t.U) {
        for (ch <- 0 until params.numChannels) {
          assert(
            ffe.module.io.out.bits(ch) === ffeGolden(ch).asTypeOf(ffe.module.io.out.bits(ch).cloneType),
            "At step %d, channel %d: Expected %d, got %d", t.asUInt, ch.asUInt, ffeGolden(ch).asTypeOf(ffe.module.io.out.bits(ch).cloneType), ffe.module.io.out.bits(ch)
          )
        }
      }
    }

    val (finishCounter, _) = Counter(true.B, timeout + 1)
    io.finished := (finishCounter === timeout.U)
  }
}

class FFETest(params: FFEParams, timeout: Int)(implicit p: Parameters) extends UnitTest(timeout) {
  val dut = Module(LazyModule(new FFETestTop(params, timeout)).module)
  dut.io.start := io.start
  io.finished := dut.io.finished
}
