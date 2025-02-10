\ Model QB_Model
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  7 x[Active_Driver_Node_0,Passive_Rider_Node_0]
   + 7 x[Active_Driver_Node_0,Passive_Rider_Node_1]
   + 3.5 x[Active_Driver_Node_0,Passive_Rider_Node_2]
   + 3.5 x[Active_Driver_Node_0,Passive_Rider_Node_3]
   + 7 x[Active_Driver_Node_0,Passive_Rider_Node_4]
   + 7 x[Active_Driver_Node_1,Passive_Rider_Node_0]
   + 7 x[Active_Driver_Node_1,Passive_Rider_Node_1]
   + 7 x[Active_Driver_Node_1,Passive_Rider_Node_2]
   + 3.5 x[Active_Driver_Node_1,Passive_Rider_Node_3]
   + 3.5 x[Active_Driver_Node_1,Passive_Rider_Node_4]
   + 3.5 x[Active_Driver_Node_2,Passive_Rider_Node_0]
   + 7 x[Active_Driver_Node_2,Passive_Rider_Node_1]
   + 7 x[Active_Driver_Node_2,Passive_Rider_Node_2]
   + 7 x[Active_Driver_Node_2,Passive_Rider_Node_3]
   + 3.5 x[Active_Driver_Node_2,Passive_Rider_Node_4]
   + 3.5 x[Active_Driver_Node_3,Passive_Rider_Node_0]
   + 3.5 x[Active_Driver_Node_3,Passive_Rider_Node_1]
   + 7 x[Active_Driver_Node_3,Passive_Rider_Node_2]
   + 7 x[Active_Driver_Node_3,Passive_Rider_Node_3]
   + 7 x[Active_Driver_Node_3,Passive_Rider_Node_4]
   + 7 x[Active_Driver_Node_4,Passive_Rider_Node_0]
   + 3.5 x[Active_Driver_Node_4,Passive_Rider_Node_1]
   + 3.5 x[Active_Driver_Node_4,Passive_Rider_Node_2]
   + 7 x[Active_Driver_Node_4,Passive_Rider_Node_3]
   + 7 x[Active_Driver_Node_4,Passive_Rider_Node_4]
Subject To
 Flow_Conservation_Active_Driver_Node_0:
   x[Active_Driver_Node_0,Passive_Rider_Node_0]
   + x[Active_Driver_Node_0,Passive_Rider_Node_1]
   + x[Active_Driver_Node_0,Passive_Rider_Node_2]
   + x[Active_Driver_Node_0,Passive_Rider_Node_3]
   + x[Active_Driver_Node_0,Passive_Rider_Node_4]
   + x_i[Active_Driver_Node_0] + x_a[Active_Driver_Node_0] = 0.5
 Flow_Conservation_Active_Driver_Node_1:
   x[Active_Driver_Node_1,Passive_Rider_Node_0]
   + x[Active_Driver_Node_1,Passive_Rider_Node_1]
   + x[Active_Driver_Node_1,Passive_Rider_Node_2]
   + x[Active_Driver_Node_1,Passive_Rider_Node_3]
   + x[Active_Driver_Node_1,Passive_Rider_Node_4]
   + x_i[Active_Driver_Node_1] + x_a[Active_Driver_Node_1] = 0.5
 Flow_Conservation_Active_Driver_Node_2:
   x[Active_Driver_Node_2,Passive_Rider_Node_0]
   + x[Active_Driver_Node_2,Passive_Rider_Node_1]
   + x[Active_Driver_Node_2,Passive_Rider_Node_2]
   + x[Active_Driver_Node_2,Passive_Rider_Node_3]
   + x[Active_Driver_Node_2,Passive_Rider_Node_4]
   + x_i[Active_Driver_Node_2] + x_a[Active_Driver_Node_2] = 0.5
 Flow_Conservation_Active_Driver_Node_3:
   x[Active_Driver_Node_3,Passive_Rider_Node_0]
   + x[Active_Driver_Node_3,Passive_Rider_Node_1]
   + x[Active_Driver_Node_3,Passive_Rider_Node_2]
   + x[Active_Driver_Node_3,Passive_Rider_Node_3]
   + x[Active_Driver_Node_3,Passive_Rider_Node_4]
   + x_i[Active_Driver_Node_3] + x_a[Active_Driver_Node_3] = 0.5
 Flow_Conservation_Active_Driver_Node_4:
   x[Active_Driver_Node_4,Passive_Rider_Node_0]
   + x[Active_Driver_Node_4,Passive_Rider_Node_1]
   + x[Active_Driver_Node_4,Passive_Rider_Node_2]
   + x[Active_Driver_Node_4,Passive_Rider_Node_3]
   + x[Active_Driver_Node_4,Passive_Rider_Node_4]
   + x_i[Active_Driver_Node_4] + x_a[Active_Driver_Node_4] = 0.5
 Passive_Flow_Conservation_Passive_Rider_Node_0:
   x[Active_Driver_Node_0,Passive_Rider_Node_0]
   + x[Active_Driver_Node_1,Passive_Rider_Node_0]
   + x[Active_Driver_Node_2,Passive_Rider_Node_0]
   + x[Active_Driver_Node_3,Passive_Rider_Node_0]
   + x[Active_Driver_Node_4,Passive_Rider_Node_0]
   + x_j[Passive_Rider_Node_0] = 0.4
 Passive_Flow_Conservation_Passive_Rider_Node_1:
   x[Active_Driver_Node_0,Passive_Rider_Node_1]
   + x[Active_Driver_Node_1,Passive_Rider_Node_1]
   + x[Active_Driver_Node_2,Passive_Rider_Node_1]
   + x[Active_Driver_Node_3,Passive_Rider_Node_1]
   + x[Active_Driver_Node_4,Passive_Rider_Node_1]
   + x_j[Passive_Rider_Node_1] = 0.4
 Passive_Flow_Conservation_Passive_Rider_Node_2:
   x[Active_Driver_Node_0,Passive_Rider_Node_2]
   + x[Active_Driver_Node_1,Passive_Rider_Node_2]
   + x[Active_Driver_Node_2,Passive_Rider_Node_2]
   + x[Active_Driver_Node_3,Passive_Rider_Node_2]
   + x[Active_Driver_Node_4,Passive_Rider_Node_2]
   + x_j[Passive_Rider_Node_2] = 0.4
 Passive_Flow_Conservation_Passive_Rider_Node_3:
   x[Active_Driver_Node_0,Passive_Rider_Node_3]
   + x[Active_Driver_Node_1,Passive_Rider_Node_3]
   + x[Active_Driver_Node_2,Passive_Rider_Node_3]
   + x[Active_Driver_Node_3,Passive_Rider_Node_3]
   + x[Active_Driver_Node_4,Passive_Rider_Node_3]
   + x_j[Passive_Rider_Node_3] = 0.4
 Passive_Flow_Conservation_Passive_Rider_Node_4:
   x[Active_Driver_Node_0,Passive_Rider_Node_4]
   + x[Active_Driver_Node_1,Passive_Rider_Node_4]
   + x[Active_Driver_Node_2,Passive_Rider_Node_4]
   + x[Active_Driver_Node_3,Passive_Rider_Node_4]
   + x[Active_Driver_Node_4,Passive_Rider_Node_4]
   + x_j[Passive_Rider_Node_4] = 0.4
 NonNegativity_Active_Driver_Node_0_Passive_Rider_Node_0:
   x[Active_Driver_Node_0,Passive_Rider_Node_0] >= 0
 NonNegativity_Active_Driver_Node_0_Passive_Rider_Node_1:
   x[Active_Driver_Node_0,Passive_Rider_Node_1] >= 0
 NonNegativity_Active_Driver_Node_0_Passive_Rider_Node_2:
   x[Active_Driver_Node_0,Passive_Rider_Node_2] >= 0
 NonNegativity_Active_Driver_Node_0_Passive_Rider_Node_3:
   x[Active_Driver_Node_0,Passive_Rider_Node_3] >= 0
 NonNegativity_Active_Driver_Node_0_Passive_Rider_Node_4:
   x[Active_Driver_Node_0,Passive_Rider_Node_4] >= 0
 NonNegativity_Active_Driver_Node_0_abandonment: x_a[Active_Driver_Node_0]
   >= 0
 NonNegativity_Active_Driver_Node_0_buffer: x_i[Active_Driver_Node_0] >= 0
 NonNegativity_Active_Driver_Node_1_Passive_Rider_Node_0:
   x[Active_Driver_Node_1,Passive_Rider_Node_0] >= 0
 NonNegativity_Active_Driver_Node_1_Passive_Rider_Node_1:
   x[Active_Driver_Node_1,Passive_Rider_Node_1] >= 0
 NonNegativity_Active_Driver_Node_1_Passive_Rider_Node_2:
   x[Active_Driver_Node_1,Passive_Rider_Node_2] >= 0
 NonNegativity_Active_Driver_Node_1_Passive_Rider_Node_3:
   x[Active_Driver_Node_1,Passive_Rider_Node_3] >= 0
 NonNegativity_Active_Driver_Node_1_Passive_Rider_Node_4:
   x[Active_Driver_Node_1,Passive_Rider_Node_4] >= 0
 NonNegativity_Active_Driver_Node_1_abandonment: x_a[Active_Driver_Node_1]
   >= 0
 NonNegativity_Active_Driver_Node_1_buffer: x_i[Active_Driver_Node_1] >= 0
 NonNegativity_Active_Driver_Node_2_Passive_Rider_Node_0:
   x[Active_Driver_Node_2,Passive_Rider_Node_0] >= 0
 NonNegativity_Active_Driver_Node_2_Passive_Rider_Node_1:
   x[Active_Driver_Node_2,Passive_Rider_Node_1] >= 0
 NonNegativity_Active_Driver_Node_2_Passive_Rider_Node_2:
   x[Active_Driver_Node_2,Passive_Rider_Node_2] >= 0
 NonNegativity_Active_Driver_Node_2_Passive_Rider_Node_3:
   x[Active_Driver_Node_2,Passive_Rider_Node_3] >= 0
 NonNegativity_Active_Driver_Node_2_Passive_Rider_Node_4:
   x[Active_Driver_Node_2,Passive_Rider_Node_4] >= 0
 NonNegativity_Active_Driver_Node_2_abandonment: x_a[Active_Driver_Node_2]
   >= 0
 NonNegativity_Active_Driver_Node_2_buffer: x_i[Active_Driver_Node_2] >= 0
 NonNegativity_Active_Driver_Node_3_Passive_Rider_Node_0:
   x[Active_Driver_Node_3,Passive_Rider_Node_0] >= 0
 NonNegativity_Active_Driver_Node_3_Passive_Rider_Node_1:
   x[Active_Driver_Node_3,Passive_Rider_Node_1] >= 0
 NonNegativity_Active_Driver_Node_3_Passive_Rider_Node_2:
   x[Active_Driver_Node_3,Passive_Rider_Node_2] >= 0
 NonNegativity_Active_Driver_Node_3_Passive_Rider_Node_3:
   x[Active_Driver_Node_3,Passive_Rider_Node_3] >= 0
 NonNegativity_Active_Driver_Node_3_Passive_Rider_Node_4:
   x[Active_Driver_Node_3,Passive_Rider_Node_4] >= 0
 NonNegativity_Active_Driver_Node_3_abandonment: x_a[Active_Driver_Node_3]
   >= 0
 NonNegativity_Active_Driver_Node_3_buffer: x_i[Active_Driver_Node_3] >= 0
 NonNegativity_Active_Driver_Node_4_Passive_Rider_Node_0:
   x[Active_Driver_Node_4,Passive_Rider_Node_0] >= 0
 NonNegativity_Active_Driver_Node_4_Passive_Rider_Node_1:
   x[Active_Driver_Node_4,Passive_Rider_Node_1] >= 0
 NonNegativity_Active_Driver_Node_4_Passive_Rider_Node_2:
   x[Active_Driver_Node_4,Passive_Rider_Node_2] >= 0
 NonNegativity_Active_Driver_Node_4_Passive_Rider_Node_3:
   x[Active_Driver_Node_4,Passive_Rider_Node_3] >= 0
 NonNegativity_Active_Driver_Node_4_Passive_Rider_Node_4:
   x[Active_Driver_Node_4,Passive_Rider_Node_4] >= 0
 NonNegativity_Active_Driver_Node_4_abandonment: x_a[Active_Driver_Node_4]
   >= 0
 NonNegativity_Active_Driver_Node_4_buffer: x_i[Active_Driver_Node_4] >= 0
 NonNegativity_Passive_Rider_Node_0_buffer: x_j[Passive_Rider_Node_0] >= 0
 NonNegativity_Passive_Rider_Node_1_buffer: x_j[Passive_Rider_Node_1] >= 0
 NonNegativity_Passive_Rider_Node_2_buffer: x_j[Passive_Rider_Node_2] >= 0
 NonNegativity_Passive_Rider_Node_3_buffer: x_j[Passive_Rider_Node_3] >= 0
 NonNegativity_Passive_Rider_Node_4_buffer: x_j[Passive_Rider_Node_4] >= 0
 Quadratic_Abandonment_Active_Driver_Node_0_Passive_Rider_Node_0:
   0.2 x[Active_Driver_Node_0,Passive_Rider_Node_0] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_0] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_1,Passive_Rider_Node_0] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_2,Passive_Rider_Node_0] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_3,Passive_Rider_Node_0] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_4,Passive_Rider_Node_0] * x_a[Active_Driver_Node_0]
   - x_j[Passive_Rider_Node_0] * x_a[Active_Driver_Node_0] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_0_Passive_Rider_Node_1:
   0.2 x[Active_Driver_Node_0,Passive_Rider_Node_1] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_1] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_1,Passive_Rider_Node_1] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_2,Passive_Rider_Node_1] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_3,Passive_Rider_Node_1] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_4,Passive_Rider_Node_1] * x_a[Active_Driver_Node_0]
   - x_j[Passive_Rider_Node_1] * x_a[Active_Driver_Node_0] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_0_Passive_Rider_Node_2:
   0.2 x[Active_Driver_Node_0,Passive_Rider_Node_2] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_2] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_1,Passive_Rider_Node_2] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_2,Passive_Rider_Node_2] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_3,Passive_Rider_Node_2] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_4,Passive_Rider_Node_2] * x_a[Active_Driver_Node_0]
   - x_j[Passive_Rider_Node_2] * x_a[Active_Driver_Node_0] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_0_Passive_Rider_Node_3:
   0.2 x[Active_Driver_Node_0,Passive_Rider_Node_3] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_3] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_1,Passive_Rider_Node_3] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_2,Passive_Rider_Node_3] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_3,Passive_Rider_Node_3] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_4,Passive_Rider_Node_3] * x_a[Active_Driver_Node_0]
   - x_j[Passive_Rider_Node_3] * x_a[Active_Driver_Node_0] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_0_Passive_Rider_Node_4:
   0.2 x[Active_Driver_Node_0,Passive_Rider_Node_4] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_4] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_1,Passive_Rider_Node_4] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_2,Passive_Rider_Node_4] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_3,Passive_Rider_Node_4] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_4,Passive_Rider_Node_4] * x_a[Active_Driver_Node_0]
   - x_j[Passive_Rider_Node_4] * x_a[Active_Driver_Node_0] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_1_Passive_Rider_Node_0:
   0.2 x[Active_Driver_Node_1,Passive_Rider_Node_0] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_0] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_1,Passive_Rider_Node_0] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_2,Passive_Rider_Node_0] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_3,Passive_Rider_Node_0] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_4,Passive_Rider_Node_0] * x_a[Active_Driver_Node_1]
   - x_j[Passive_Rider_Node_0] * x_a[Active_Driver_Node_1] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_1_Passive_Rider_Node_1:
   0.2 x[Active_Driver_Node_1,Passive_Rider_Node_1] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_1] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_1,Passive_Rider_Node_1] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_2,Passive_Rider_Node_1] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_3,Passive_Rider_Node_1] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_4,Passive_Rider_Node_1] * x_a[Active_Driver_Node_1]
   - x_j[Passive_Rider_Node_1] * x_a[Active_Driver_Node_1] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_1_Passive_Rider_Node_2:
   0.2 x[Active_Driver_Node_1,Passive_Rider_Node_2] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_2] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_1,Passive_Rider_Node_2] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_2,Passive_Rider_Node_2] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_3,Passive_Rider_Node_2] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_4,Passive_Rider_Node_2] * x_a[Active_Driver_Node_1]
   - x_j[Passive_Rider_Node_2] * x_a[Active_Driver_Node_1] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_1_Passive_Rider_Node_3:
   0.2 x[Active_Driver_Node_1,Passive_Rider_Node_3] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_3] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_1,Passive_Rider_Node_3] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_2,Passive_Rider_Node_3] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_3,Passive_Rider_Node_3] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_4,Passive_Rider_Node_3] * x_a[Active_Driver_Node_1]
   - x_j[Passive_Rider_Node_3] * x_a[Active_Driver_Node_1] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_1_Passive_Rider_Node_4:
   0.2 x[Active_Driver_Node_1,Passive_Rider_Node_4] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_4] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_1,Passive_Rider_Node_4] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_2,Passive_Rider_Node_4] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_3,Passive_Rider_Node_4] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_4,Passive_Rider_Node_4] * x_a[Active_Driver_Node_1]
   - x_j[Passive_Rider_Node_4] * x_a[Active_Driver_Node_1] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_2_Passive_Rider_Node_0:
   0.2 x[Active_Driver_Node_2,Passive_Rider_Node_0] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_0] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_1,Passive_Rider_Node_0] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_2,Passive_Rider_Node_0] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_3,Passive_Rider_Node_0] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_4,Passive_Rider_Node_0] * x_a[Active_Driver_Node_2]
   - x_j[Passive_Rider_Node_0] * x_a[Active_Driver_Node_2] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_2_Passive_Rider_Node_1:
   0.2 x[Active_Driver_Node_2,Passive_Rider_Node_1] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_1] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_1,Passive_Rider_Node_1] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_2,Passive_Rider_Node_1] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_3,Passive_Rider_Node_1] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_4,Passive_Rider_Node_1] * x_a[Active_Driver_Node_2]
   - x_j[Passive_Rider_Node_1] * x_a[Active_Driver_Node_2] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_2_Passive_Rider_Node_2:
   0.2 x[Active_Driver_Node_2,Passive_Rider_Node_2] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_2] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_1,Passive_Rider_Node_2] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_2,Passive_Rider_Node_2] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_3,Passive_Rider_Node_2] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_4,Passive_Rider_Node_2] * x_a[Active_Driver_Node_2]
   - x_j[Passive_Rider_Node_2] * x_a[Active_Driver_Node_2] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_2_Passive_Rider_Node_3:
   0.2 x[Active_Driver_Node_2,Passive_Rider_Node_3] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_3] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_1,Passive_Rider_Node_3] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_2,Passive_Rider_Node_3] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_3,Passive_Rider_Node_3] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_4,Passive_Rider_Node_3] * x_a[Active_Driver_Node_2]
   - x_j[Passive_Rider_Node_3] * x_a[Active_Driver_Node_2] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_2_Passive_Rider_Node_4:
   0.2 x[Active_Driver_Node_2,Passive_Rider_Node_4] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_4] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_1,Passive_Rider_Node_4] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_2,Passive_Rider_Node_4] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_3,Passive_Rider_Node_4] * x_a[Active_Driver_Node_2]
   - x[Active_Driver_Node_4,Passive_Rider_Node_4] * x_a[Active_Driver_Node_2]
   - x_j[Passive_Rider_Node_4] * x_a[Active_Driver_Node_2] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_3_Passive_Rider_Node_0:
   0.2 x[Active_Driver_Node_3,Passive_Rider_Node_0] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_0] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_1,Passive_Rider_Node_0] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_2,Passive_Rider_Node_0] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_3,Passive_Rider_Node_0] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_4,Passive_Rider_Node_0] * x_a[Active_Driver_Node_3]
   - x_j[Passive_Rider_Node_0] * x_a[Active_Driver_Node_3] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_3_Passive_Rider_Node_1:
   0.2 x[Active_Driver_Node_3,Passive_Rider_Node_1] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_1] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_1,Passive_Rider_Node_1] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_2,Passive_Rider_Node_1] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_3,Passive_Rider_Node_1] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_4,Passive_Rider_Node_1] * x_a[Active_Driver_Node_3]
   - x_j[Passive_Rider_Node_1] * x_a[Active_Driver_Node_3] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_3_Passive_Rider_Node_2:
   0.2 x[Active_Driver_Node_3,Passive_Rider_Node_2] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_2] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_1,Passive_Rider_Node_2] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_2,Passive_Rider_Node_2] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_3,Passive_Rider_Node_2] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_4,Passive_Rider_Node_2] * x_a[Active_Driver_Node_3]
   - x_j[Passive_Rider_Node_2] * x_a[Active_Driver_Node_3] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_3_Passive_Rider_Node_3:
   0.2 x[Active_Driver_Node_3,Passive_Rider_Node_3] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_3] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_1,Passive_Rider_Node_3] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_2,Passive_Rider_Node_3] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_3,Passive_Rider_Node_3] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_4,Passive_Rider_Node_3] * x_a[Active_Driver_Node_3]
   - x_j[Passive_Rider_Node_3] * x_a[Active_Driver_Node_3] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_3_Passive_Rider_Node_4:
   0.2 x[Active_Driver_Node_3,Passive_Rider_Node_4] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_4] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_1,Passive_Rider_Node_4] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_2,Passive_Rider_Node_4] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_3,Passive_Rider_Node_4] * x_a[Active_Driver_Node_3]
   - x[Active_Driver_Node_4,Passive_Rider_Node_4] * x_a[Active_Driver_Node_3]
   - x_j[Passive_Rider_Node_4] * x_a[Active_Driver_Node_3] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_4_Passive_Rider_Node_0:
   0.2 x[Active_Driver_Node_4,Passive_Rider_Node_0] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_0] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_1,Passive_Rider_Node_0] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_2,Passive_Rider_Node_0] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_3,Passive_Rider_Node_0] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_4,Passive_Rider_Node_0] * x_a[Active_Driver_Node_4]
   - x_j[Passive_Rider_Node_0] * x_a[Active_Driver_Node_4] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_4_Passive_Rider_Node_1:
   0.2 x[Active_Driver_Node_4,Passive_Rider_Node_1] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_1] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_1,Passive_Rider_Node_1] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_2,Passive_Rider_Node_1] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_3,Passive_Rider_Node_1] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_4,Passive_Rider_Node_1] * x_a[Active_Driver_Node_4]
   - x_j[Passive_Rider_Node_1] * x_a[Active_Driver_Node_4] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_4_Passive_Rider_Node_2:
   0.2 x[Active_Driver_Node_4,Passive_Rider_Node_2] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_2] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_1,Passive_Rider_Node_2] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_2,Passive_Rider_Node_2] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_3,Passive_Rider_Node_2] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_4,Passive_Rider_Node_2] * x_a[Active_Driver_Node_4]
   - x_j[Passive_Rider_Node_2] * x_a[Active_Driver_Node_4] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_4_Passive_Rider_Node_3:
   0.2 x[Active_Driver_Node_4,Passive_Rider_Node_3] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_3] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_1,Passive_Rider_Node_3] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_2,Passive_Rider_Node_3] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_3,Passive_Rider_Node_3] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_4,Passive_Rider_Node_3] * x_a[Active_Driver_Node_4]
   - x_j[Passive_Rider_Node_3] * x_a[Active_Driver_Node_4] ] <= 0
 Quadratic_Abandonment_Active_Driver_Node_4_Passive_Rider_Node_4:
   0.2 x[Active_Driver_Node_4,Passive_Rider_Node_4] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_4] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_1,Passive_Rider_Node_4] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_2,Passive_Rider_Node_4] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_3,Passive_Rider_Node_4] * x_a[Active_Driver_Node_4]
   - x[Active_Driver_Node_4,Passive_Rider_Node_4] * x_a[Active_Driver_Node_4]
   - x_j[Passive_Rider_Node_4] * x_a[Active_Driver_Node_4] ] <= 0
Bounds
End
