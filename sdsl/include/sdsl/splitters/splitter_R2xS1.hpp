#ifndef _SDSL_VOXEL_SPLITTER_R2XS1_HPP
#define _SDSL_VOXEL_SPLITTER_R2XS1_HPP
#pragma once

#include "sdsl/splitters/splitter.hpp"
#include "sdsl/configurations/config_R2xS1.hpp"

namespace sdsl {

    template<typename FT>
    class Splitter_R2xS1 {
    public:
        Splitter_R2xS1() : numSplitX(2), numSplitY(2), numSplitR(2), incrementCount(0) {}
        Splitter_R2xS1(int x, int y, int r) : numSplitX(x), numSplitY(y), numSplitR(r), incrementCount(0) {}

        void inc() { incrementCount++; }

        void operator()(Voxel<R2xS1<FT>>& v, std::vector<Voxel<R2xS1<FT>>>& out) {
            std::vector<Voxel<R2xS1<FT>>> queue1, queue2; // And we swap between those to avoid unneseecary copying
            queue1.push_back(v);

            // Split by X
            for (Voxel<R2xS1<FT>>& voxel : queue1) {
                FT left = v.bottomLeft().getX(); 
                FT right = v.topRight().getX();
                for (int i = 0; i < numSplitX; i++) {
                    FT a = left + ((double)i / (double)numSplitX) * (right - left);
                    FT b = left + ((double)(i+1) / (double)numSplitX) * (right - left);
                    queue2.push_back(Voxel(
                        R2xS1(a, voxel.bottomLeft().getY(), voxel.bottomLeft().getR()),
                        R2xS1(b, voxel.topRight().getY(), voxel.topRight().getR())
                    ));
                }
            }

            // Split by Y
            queue1.clear();
            for (Voxel<R2xS1<FT>>& voxel: queue2) {
                FT left = v.bottomLeft().getY();
                FT right = v.topRight().getY();
                for (int i = 0; i < numSplitY; i++) {
                    FT a = left + ((double)i / (double)numSplitY) * (right - left);
                    FT b = left + ((double)(i+1) / (double)numSplitY) * (right - left);
                    queue1.push_back(Voxel(
                        R2xS1(voxel.bottomLeft().getX(), a, voxel.bottomLeft().getR()),
                        R2xS1(voxel.topRight().getX(), b, voxel.topRight().getR())
                    ));
                }
            }

            // Split by R
            // Notice that since this is last dimension, we append directly into output
            queue2.clear();
            for (Voxel<R2xS1<FT>>& voxel: queue1) {
                FT left = v.bottomLeft().getR();
                FT right = v.topRight().getR();
                for (int i = 0; i < numSplitR; i++) {
                    FT a = left + ((double)i / (double)numSplitR) * (right - left);
                    FT b = left + ((double)(i+1) / (double)numSplitR) * (right - left);
                    out.push_back(Voxel(
                        R2xS1(voxel.bottomLeft().getX(), voxel.bottomLeft().getY(), a),
                        R2xS1(voxel.topRight().getX(), voxel.topRight().getY(), b)
                    ));
                }
            }
        }

    private:
        int numSplitX, numSplitY, numSplitR;
        int incrementCount;
    };

}

#endif