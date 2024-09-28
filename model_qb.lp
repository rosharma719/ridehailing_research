\ Model QB_Model
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  8 x[Active_Driver_Node_0,Passive_Rider_Node_0]
   + 7 x[Active_Driver_Node_0,Passive_Rider_Node_1]
   + 7 x[Active_Driver_Node_1,Passive_Rider_Node_0]
   + 8 x[Active_Driver_Node_1,Passive_Rider_Node_1]
Subject To
 Flow_Constraint_Active_Driver_Node_0:
   x[Active_Driver_Node_0,Passive_Rider_Node_0]
   + x[Active_Driver_Node_0,Passive_Rider_Node_1]
   + xr[Active_Driver_Node_0] = 0.3
 Flow_Constraint_Active_Driver_Node_1:
   x[Active_Driver_Node_1,Passive_Rider_Node_0]
   + x[Active_Driver_Node_1,Passive_Rider_Node_1]
   + xr[Active_Driver_Node_1] = 0.3
 Equation_Constraint_Active_Driver_Node_0_Passive_Rider_Node_0:
   0.2 xr[Active_Driver_Node_0] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_0] * xda[Passive_Rider_Node_0]
   - xr[Active_Driver_Node_0] * xda[Passive_Rider_Node_0] ] = 0
 Equation_Constraint_Active_Driver_Node_1_Passive_Rider_Node_0:
   0.2 xr[Active_Driver_Node_1] + [
   - x[Active_Driver_Node_1,Passive_Rider_Node_0] * xda[Passive_Rider_Node_0]
   - xr[Active_Driver_Node_1] * xda[Passive_Rider_Node_0] ] = 0
 Equation_Constraint_Active_Driver_Node_0_Passive_Rider_Node_1:
   0.2 xr[Active_Driver_Node_0] + [
   - x[Active_Driver_Node_0,Passive_Rider_Node_1] * xda[Passive_Rider_Node_1]
   - xr[Active_Driver_Node_0] * xda[Passive_Rider_Node_1] ] = 0
 Equation_Constraint_Active_Driver_Node_1_Passive_Rider_Node_1:
   0.2 xr[Active_Driver_Node_1] + [
   - x[Active_Driver_Node_1,Passive_Rider_Node_1] * xda[Passive_Rider_Node_1]
   - xr[Active_Driver_Node_1] * xda[Passive_Rider_Node_1] ] = 0
Bounds
End
