\ Model QB_Model
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x815c568492b5906e
Maximize
  2 x[Active_Driver_Node_0,Passive_Rider_Node_0]
   + 2 x[Active_Driver_Node_0,Passive_Rider_Node_1]
   + 2 x[Active_Driver_Node_0,Passive_Rider_Node_2]
   + 2 x[Active_Driver_Node_1,Passive_Rider_Node_0]
   + 2 x[Active_Driver_Node_1,Passive_Rider_Node_1]
   + 100 x[Active_Driver_Node_1,Passive_Rider_Node_2]
   + 2 x[Active_Driver_Node_2,Passive_Rider_Node_0]
   + 2 x[Active_Driver_Node_2,Passive_Rider_Node_1]
   + 2 x[Active_Driver_Node_2,Passive_Rider_Node_2]
Subject To
 Flow_Conservation_Active_Driver_Node_0:
   x[Active_Driver_Node_0,Passive_Rider_Node_0]
   + x[Active_Driver_Node_0,Passive_Rider_Node_1]
   + x[Active_Driver_Node_0,Passive_Rider_Node_2]
   + x_i[Active_Driver_Node_0] + x_a[Active_Driver_Node_0] = 0.2
 Flow_Conservation_Active_Driver_Node_1:
   x[Active_Driver_Node_1,Passive_Rider_Node_0]
   + x[Active_Driver_Node_1,Passive_Rider_Node_1]
   + x[Active_Driver_Node_1,Passive_Rider_Node_2]
   + x_i[Active_Driver_Node_1] + x_a[Active_Driver_Node_1] = 0.5
 Flow_Conservation_Active_Driver_Node_2:
   x[Active_Driver_Node_2,Passive_Rider_Node_0]
   + x[Active_Driver_Node_2,Passive_Rider_Node_1]
   + x[Active_Driver_Node_2,Passive_Rider_Node_2]
   + x_i[Active_Driver_Node_2] + x_a[Active_Driver_Node_2] = 0.1
 Passive_Flow_Conservation_Passive_Rider_Node_0:
   x[Active_Driver_Node_0,Passive_Rider_Node_0]
   + x[Active_Driver_Node_1,Passive_Rider_Node_0]
   + x[Active_Driver_Node_2,Passive_Rider_Node_0]
   + x_j[Passive_Rider_Node_0] = 0.8
 Passive_Flow_Conservation_Passive_Rider_Node_1:
   x[Active_Driver_Node_0,Passive_Rider_Node_1]
   + x[Active_Driver_Node_1,Passive_Rider_Node_1]
   + x[Active_Driver_Node_2,Passive_Rider_Node_1]
   + x_j[Passive_Rider_Node_1] = 0.1
 Passive_Flow_Conservation_Passive_Rider_Node_2:
   x[Active_Driver_Node_0,Passive_Rider_Node_2]
   + x[Active_Driver_Node_1,Passive_Rider_Node_2]
   + x[Active_Driver_Node_2,Passive_Rider_Node_2]
   + x_j[Passive_Rider_Node_2] = 0.1
 NonNegativity_Active_Driver_Node_0_Passive_Rider_Node_0:
   x[Active_Driver_Node_0,Passive_Rider_Node_0] >= 0
 NonNegativity_Active_Driver_Node_0_Passive_Rider_Node_1:
   x[Active_Driver_Node_0,Passive_Rider_Node_1] >= 0
 NonNegativity_Active_Driver_Node_0_Passive_Rider_Node_2:
   x[Active_Driver_Node_0,Passive_Rider_Node_2] >= 0
 NonNegativity_Active_Driver_Node_0_abandonment: x_a[Active_Driver_Node_0]
   >= 0
 NonNegativity_Active_Driver_Node_0_buffer: x_i[Active_Driver_Node_0] >= 0
 NonNegativity_Active_Driver_Node_1_Passive_Rider_Node_0:
   x[Active_Driver_Node_1,Passive_Rider_Node_0] >= 0
 NonNegativity_Active_Driver_Node_1_Passive_Rider_Node_1:
   x[Active_Driver_Node_1,Passive_Rider_Node_1] >= 0
 NonNegativity_Active_Driver_Node_1_Passive_Rider_Node_2:
   x[Active_Driver_Node_1,Passive_Rider_Node_2] >= 0
 NonNegativity_Active_Driver_Node_1_abandonment: x_a[Active_Driver_Node_1]
   >= 0
 NonNegativity_Active_Driver_Node_1_buffer: x_i[Active_Driver_Node_1] >= 0
 NonNegativity_Active_Driver_Node_2_Passive_Rider_Node_0:
   x[Active_Driver_Node_2,Passive_Rider_Node_0] >= 0
 NonNegativity_Active_Driver_Node_2_Passive_Rider_Node_1:
   x[Active_Driver_Node_2,Passive_Rider_Node_1] >= 0
 NonNegativity_Active_Driver_Node_2_Passive_Rider_Node_2:
   x[Active_Driver_Node_2,Passive_Rider_Node_2] >= 0
 NonNegativity_Active_Driver_Node_2_abandonment: x_a[Active_Driver_Node_2]
   >= 0
 NonNegativity_Active_Driver_Node_2_buffer: x_i[Active_Driver_Node_2] >= 0
 NonNegativity_Passive_Rider_Node_0_buffer: x_j[Passive_Rider_Node_0] >= 0
 NonNegativity_Passive_Rider_Node_1_buffer: x_j[Passive_Rider_Node_1] >= 0
 NonNegativity_Passive_Rider_Node_2_buffer: x_j[Passive_Rider_Node_2] >= 0
 Quadratic_Abandonment_Active_Driver_Node_0_Passive_Rider_Node_0:
   x[Active_Driver_Node_0,Passive_Rider_Node_0] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_0] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_1,Passive_Rider_Node_0] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_2,Passive_Rider_Node_0] * x_a[Active_Driver_Node_0]
   - x_j[Passive_Rider_Node_0] * x_a[Active_Driver_Node_0] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_0_Passive_Rider_Node_1:
   x[Active_Driver_Node_0,Passive_Rider_Node_1] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_1] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_1,Passive_Rider_Node_1] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_2,Passive_Rider_Node_1] * x_a[Active_Driver_Node_0]
   - x_j[Passive_Rider_Node_1] * x_a[Active_Driver_Node_0] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_0_Passive_Rider_Node_2:
   x[Active_Driver_Node_0,Passive_Rider_Node_2] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_2] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_1,Passive_Rider_Node_2] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_2,Passive_Rider_Node_2] * x_a[Active_Driver_Node_0]
   - x_j[Passive_Rider_Node_2] * x_a[Active_Driver_Node_0] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_1_Passive_Rider_Node_0:
   0.5 x[Active_Driver_Node_1,Passive_Rider_Node_0] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_0] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_1,Passive_Rider_Node_0] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_2,Passive_Rider_Node_0] * x_a[Active_Driver_Node_1]
   - x_j[Passive_Rider_Node_0] * x_a[Active_Driver_Node_1] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_1_Passive_Rider_Node_1:
   0.5 x[Active_Driver_Node_1,Passive_Rider_Node_1] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_1] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_1,Passive_Rider_Node_1] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_2,Passive_Rider_Node_1] * x_a[Active_Driver_Node_1]
   - x_j[Passive_Rider_Node_1] * x_a[Active_Driver_Node_1] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_1_Passive_Rider_Node_2:
   0.5 x[Active_Driver_Node_1,Passive_Rider_Node_2] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_2] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_1,Passive_Rider_Node_2] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_2,Passive_Rider_Node_2] * x_a[Active_Driver_Node_1]
   - x_j[Passive_Rider_Node_2] * x_a[Active_Driver_Node_1] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_2_Passive_Rider_Node_0:
   0.3 x[Active_Driver_Node_2,Passive_Rider_Node_0] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_0] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_1,Passive_Rider_Node_0] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_2,Passive_Rider_Node_0] * x_a[Active_Driver_Node_2]
   - x_j[Passive_Rider_Node_0] * x_a[Active_Driver_Node_2] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_2_Passive_Rider_Node_1:
   0.3 x[Active_Driver_Node_2,Passive_Rider_Node_1] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_1] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_1,Passive_Rider_Node_1] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_2,Passive_Rider_Node_1] * x_a[Active_Driver_Node_2]
   - x_j[Passive_Rider_Node_1] * x_a[Active_Driver_Node_2] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_2_Passive_Rider_Node_2:
   0.3 x[Active_Driver_Node_2,Passive_Rider_Node_2] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_2] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_1,Passive_Rider_Node_2] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_2,Passive_Rider_Node_2] * x_a[Active_Driver_Node_2]
   - x_j[Passive_Rider_Node_2] * x_a[Active_Driver_Node_2] ] <= 0
Bounds
End
