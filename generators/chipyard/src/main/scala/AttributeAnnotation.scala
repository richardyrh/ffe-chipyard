
package firrtl {
  import firrtl.annotations._

  /** used by [[addAttribute]] to add SystemVerilog attributes to a [[Target]].
   *
   * The class name `firrtl.AttributeAnnotation` is recognized by firtool
   *
   * @param target
   * @param description
   */
  case class AttributeAnnotation(target: Target, description: String) extends SingleTargetAnnotation[Target] {
    def targets = Seq(target)
    def duplicate(n: Target) = this.copy(n, description)

    override def serialize: String = s"AttributeAnnotation(${target.serialize}, $description)"
  }

}


package chisel3.util {
  object addAttribute { // scalastyle:ignore object.name
    def apply[T <: chisel3.InstanceId](instance: T, attributes: (String, Any)*): T = {

      for ((attr, value) <- attributes) {
        apply(instance, attr, value)
      }
      instance
    }

    def apply[T <: chisel3.InstanceId](inst: T, attribute: String, value: Any): T = {
      chisel3.experimental.annotate(
        new chisel3.experimental.ChiselAnnotation {
          val valueStr = value match {
            case _: String => s"\"$value\""
            case _ => value.toString
          }
          override def toFirrtl = new firrtl.AttributeAnnotation(inst.toTarget, s"$attribute = $valueStr")

        }
      )
      inst
    }
  }

}
