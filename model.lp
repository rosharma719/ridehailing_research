\ Model RB_Model
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  2 x[Active_Driver_Node_0,Passive_Rider_Node_0]
   + x[Active_Driver_Node_0,Passive_Rider_Node_1]
   + x[Active_Driver_Node_1,Passive_Rider_Node_0]
   + 2 x[Active_Driver_Node_1,Passive_Rider_Node_1]
Subject To
 Flow_Constraint_Active_Driver_Node_0:
   x[Active_Driver_Node_0,Passive_Rider_Node_0]
   + x[Active_Driver_Node_0,Passive_Rider_Node_1]
   + x_a[Active_Driver_Node_0] = 0.3
 Flow_Constraint_Active_Driver_Node_1:
   x[Active_Driver_Node_1,Passive_Rider_Node_0]
   + x[Active_Driver_Node_1,Passive_Rider_Node_1]
   + x_a[Active_Driver_Node_1] = 0.3
 Abandonment_Constraint_Active_Driver_Node_0_Passive_Rider_Node_0:
   0.2 x[Active_Driver_Node_0,Passive_Rider_Node_0]
   - 0.4 x_a[Active_Driver_Node_0] <= 0
 Abandonment_Constraint_Active_Driver_Node_0_Passive_Rider_Node_1:
   0.2 x[Active_Driver_Node_0,Passive_Rider_Node_1]
   - 0.4 x_a[Active_Driver_Node_0] <= 0
 Abandonment_Constraint_Active_Driver_Node_1_Passive_Rider_Node_0:
   0.2 x[Active_Driver_Node_1,Passive_Rider_Node_0]
   - 0.4 x_a[Active_Driver_Node_1] <= 0
 Abandonment_Constraint_Active_Driver_Node_1_Passive_Rider_Node_1:
   0.2 x[Active_Driver_Node_1,Passive_Rider_Node_1]
   - 0.4 x_a[Active_Driver_Node_1] <= 0
Bounds
End
