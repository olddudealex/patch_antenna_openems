import CSXCAD
import os

def build_csx(epsilon, kappa):
    CSX = CSXCAD.ContinuousStructure()

    #######################################################################################################################################
    # MATERIALS AND GEOMETRY
    #######################################################################################################################################
    currDir = "line_array_openEMS_simulation"

    materialList = {}

    ## MATERIAL - PEC
    materialList['PEC'] = CSX.AddMetal('PEC')

    materialList['PEC'].AddPolyhedronReader(os.path.join(currDir, 'top_gen_model.stl'), priority=10).ReadFile()

    ## MATERIAL - GND
    materialList['GND'] = CSX.AddMetal('GND')

    materialList['GND'].AddPolyhedronReader(os.path.join(currDir, 'bottom_gen_model.stl'), priority=10).ReadFile()

    ## MATERIAL - SUBSTRATE
    materialList['SUBSTRATE'] = CSX.AddMaterial('SUBSTRATE')

    materialList['SUBSTRATE'].SetMaterialProperty(epsilon=epsilon, mue=1.0, kappa=kappa, sigma=0.0)
    materialList['SUBSTRATE'].AddPolyhedronReader(os.path.join(currDir, 'substrate_gen_model.stl'),
                                                  priority=0).ReadFile()

    ## MATERIAL - AIR
    #materialList['AIR'] = CSX.AddMaterial('AIR')

    #materialList['AIR'].SetMaterialProperty(epsilon=1.0, mue=1.0, kappa=0.0, sigma=0.0)
    #materialList['AIR'].AddPolyhedronReader(os.path.join(currDir, 'air_gen_model.stl'), priority=9900).ReadFile()

    return CSX
