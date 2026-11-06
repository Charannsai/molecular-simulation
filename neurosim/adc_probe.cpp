// Probe NeuroSim V1.4's own SAR ADC model: energy and area of one converter vs bits and node.
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "SarADC.h"
#include "Definition.h"
using namespace std;

int main() {
    int nodes[] = {22, 14, 7};
    double colRes[] = {1e30, 781.0};  // ADC alone; a 512-row column at 10 uS window, mean 2.5 uS per active cell
    cout << "node,roadmap,bits,colRes_ohm,energy_pJ,area_um2,latency_ns" << endl;
    for (int node : nodes) {
        for (int rm = (node <= 14 ? 2 : 1); rm <= 2; rm++) {
            param->technode = node;
            param->deviceroadmap = rm;
            inputParameter.transistorType = conventional;
            inputParameter.deviceRoadmap = (rm == 1) ? HP : LSTP;
            inputParameter.temperature = 300;
            inputParameter.processNode = node;
            tech.Initialize(node, inputParameter.deviceRoadmap, inputParameter.transistorType);
            cell.memCellType = Type::RRAM;
            for (int bits = 5; bits <= 12; bits++) {
                for (double r : colRes) {
                    SarADC adc(inputParameter, tech, cell);
                    adc.Initialize(1, (int) pow(2, bits), 1e9, 1);
                    adc.CalculateUnitArea();
                    adc.CalculateArea(0, 1e-3, NONE);
                    vector<double> cr(1, r);
                    adc.CalculatePower(cr, 1);
                    adc.CalculateLatency(1);
                    cout << node << "," << (rm == 1 ? "HP" : "LSTP") << "," << bits << "," << (r > 1e20 ? 0 : r) << ","
                         << adc.readDynamicEnergy * 1e12 << "," << adc.area * 1e12 << "," << adc.readLatency * 1e9 << endl;
                }
            }
        }
    }
    return 0;
}
