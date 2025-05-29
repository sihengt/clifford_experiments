import torch

def fk_single_step(front_ang,rear_ang,arcLen,wheelBase):
    trig_term_1 = rear_ang.cos()*front_ang.sin()+front_ang.cos()*rear_ang.sin()
    trig_term_2 = front_ang.cos()*rear_ang.sin()-rear_ang.cos()*front_ang.sin()
    cen_x = wheelBase/2.0*trig_term_2/trig_term_1
    cen_y = rear_ang.cos()*front_ang.cos()*wheelBase/(front_ang.cos()*rear_ang.sin()+rear_ang.cos()*front_ang.sin())
    turnCenter = torch.cat((cen_x.unsqueeze(-1),cen_y.unsqueeze(-1)),dim=-1)
    turnRadius = torch.norm(turnCenter,dim=-1)
    arcAng = arcLen/turnRadius*torch.sign(cen_y)
    relX = -cen_x*arcAng.cos() + cen_y*arcAng.sin() + cen_x
    relY = -cen_x*arcAng.sin() - cen_y*arcAng.cos() + cen_y
    rel_nonstrafe = torch.cat((relX.unsqueeze(-1),
                    relY.unsqueeze(-1),
                    arcAng.unsqueeze(-1)),dim=-1)

    # handle cases for strafing
    travelAng = (front_ang-rear_ang)/2.0
    relX = arcLen*travelAng.cos()
    relY = arcLen*travelAng.sin()
    relAng = torch.zeros_like(travelAng)
    rel_strafe = torch.cat((relX.unsqueeze(-1),
                            relY.unsqueeze(-1),
                            relAng.unsqueeze(-1)),dim=-1)
    rel = torch.where(torch.isnan(rel_nonstrafe),rel_strafe,rel_nonstrafe)
    return rel

def calcSteerAngle(relX,relY):
    steer_ang = torch.atan2(relX,relY)
    steer_ang = torch.where(steer_ang.abs() > torch.pi/2.0,
                            steer_ang-steer_ang.sign()*torch.pi,
                            steer_ang)
    #if torch.abs(steer_ang) > torch.pi/2.0:
    #    steer_ang -= torch.sign(steer_ang)*torch.pi
    return steer_ang

def ik_single_step(relTrans,wheelBase,forward_only=False):
    d = torch.norm(relTrans[...,:2],dim=-1)
    turnRadius = (d/(2.0*torch.sin(relTrans[...,2]/2.0))).clamp(max=10000)
    ang = torch.atan2(relTrans[...,1],relTrans[...,0]) + (torch.pi/2.0 - relTrans[...,2]/2.0)
    cen_x = turnRadius*ang.cos()
    cen_y = turnRadius*ang.sin()
    front_ang = calcSteerAngle(wheelBase/2.0 - cen_x, cen_y)
    rear_ang = -calcSteerAngle(-wheelBase/2.0 - cen_x, cen_y)
    # calculate drive direction
    crossProd_dest2cen = relTrans[...,0]*cen_y - relTrans[...,1]*cen_x
    driveDir = torch.sign(crossProd_dest2cen*cen_y)
    driveDir = torch.where(driveDir == 0,torch.ones_like(driveDir),driveDir)
    arcAng = 2.0*torch.asin(d/2.0/turnRadius)*driveDir #*torch.sign(relTrans[...,0])
    if forward_only:
        arcAng = torch.where(arcAng<0,arcAng+2*torch.pi,arcAng)
    arcLen = arcAng*turnRadius
    return front_ang,rear_ang,arcLen
    
def ik_loc(relLoc,wheelBase,forward_only=False):
    d = torch.norm(relLoc[...,:2],dim=-1)
    turnRadius = d**2/2.0/torch.abs(relLoc[...,1])
    cen_y = turnRadius*torch.sign(relLoc[...,1])
    cen_x = torch.zeros_like(cen_y)
    
    front_ang = calcSteerAngle(wheelBase/2.0 - cen_x, cen_y)
    rear_ang = -calcSteerAngle(-wheelBase/2.0 - cen_x, cen_y)

    # calculate drive direction
    crossProd_dest2cen = relLoc[...,0]*cen_y - relLoc[...,1]*cen_x
    driveDir = torch.sign(crossProd_dest2cen*cen_y)
    driveDir = torch.where(driveDir == 0,torch.ones_like(driveDir),driveDir)
    arcAng = 2.0*torch.asin(d/2.0/turnRadius)*driveDir
    if forward_only:
        arcAng = torch.where(arcAng<0,arcAng+2*torch.pi,arcAng)
    arcLen = arcAng*turnRadius
    return front_ang,rear_ang,arcLen

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    stepSize = 0.1
