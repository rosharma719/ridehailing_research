\ Model QB_Model
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  3 x[Active_Driver_Node_0,Passive_Rider_Node_0]
   + 2 x[Active_Driver_Node_0,Passive_Rider_Node_1]
   + 2 x[Active_Driver_Node_1,Passive_Rider_Node_0]
   + 3 x[Active_Driver_Node_1,Passive_Rider_Node_1]
Subject To
 Flow_Conservation_Active_Driver_Node_0:
   x[Active_Driver_Node_0,Passive_Rider_Node_0]
   + x[Active_Driver_Node_0,Passive_Rider_Node_1]
   + x_a[Active_Driver_Node_0] = 0.3
 Flow_Conservation_Active_Driver_Node_1:
   x[Active_Driver_Node_1,Passive_Rider_Node_0]
   + x[Active_Driver_Node_1,Passive_Rider_Node_1]
   + x_a[Active_Driver_Node_1] = 0.3
 Abandonment_Active_Driver_Node_0_Passive_Rider_Node_0:
   0.2 x[Active_Driver_Node_0,Passive_Rider_Node_0] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_0] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_1,Passive_Rider_Node_0] * x_a[Active_Driver_Node_0]
   - x_j[Passive_Rider_Node_0] * x_a[Active_Driver_Node_0] ] <= 0
 Abandonment_Active_Driver_Node_0_Passive_Rider_Node_1:
   0.2 x[Active_Driver_Node_0,Passive_Rider_Node_1] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_1] * x_a[Active_Driver_Node_0]
   - x[Active_Driver_Node_1,Passive_Rider_Node_1] * x_a[Active_Driver_Node_0]
   - x_j[Passive_Rider_Node_1] * x_a[Active_Driver_Node_0] ] <= 0
 Abandonment_Active_Driver_Node_1_Passive_Rider_Node_0:
   0.2 x[Active_Driver_Node_1,Passive_Rider_Node_0] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_0] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_1,Passive_Rider_Node_0] * x_a[Active_Driver_Node_1]
   - x_j[Passive_Rider_Node_0] * x_a[Active_Driver_Node_1] ] <= 0
 Abandonment_Active_Driver_Node_1_Passive_Rider_Node_1:
   0.2 x[Active_Driver_Node_1,Passive_Rider_Node_1] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_1] * x_a[Active_Driver_Node_1]
   - x[Active_Driver_Node_1,Passive_Rider_Node_1] * x_a[Active_Driver_Node_1]
   - x_j[Passive_Rider_Node_1] * x_a[Active_Driver_Node_1] ] <= 0
Bounds
End
