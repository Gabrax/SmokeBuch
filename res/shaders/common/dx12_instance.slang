// GLM uploads four consecutive columns. Explicit vectors keep the structured
// buffer ABI independent of Slang/FXC's matrix storage convention.
struct InstanceTransform
{
  float4 column0;
  float4 column1;
  float4 column2;
  float4 column3;
};

float4x4 InstanceMatrix(InstanceTransform transform)
{
  return transpose(float4x4(transform.column0, transform.column1,
    transform.column2, transform.column3));
}
