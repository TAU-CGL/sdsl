#ifndef _SDSL_VOXEL_SPLITTER_R3XS1_HPP
#define _SDSL_VOXEL_SPLITTER_R3XS1_HPP
#pragma once

#include "sdsl/splitters/splitter.hpp"
#include "sdsl/configurations/config_R3xS1.hpp"

#include <fmt/core.h>

namespace sdsl {

    template<typename FT>
    class Splitter_R3xS1 {
    public:
        Splitter_R3xS1() : numSplitX(2), numSplitY(2), numSplitZ(2), numSplitR(2), incrementCount(0) {}
        Splitter_R3xS1(int x, int y, int z, int r) : numSplitX(x), numSplitY(y), numSplitZ(z), numSplitR(r), incrementCount(0) {}

        virtual void inc() { incrementCount++; }

        void operator()(Voxel<R3xS1<FT>>& v, std::vector<Voxel<R3xS1<FT>>>& out) {
            std::vector<Voxel<R3xS1<FT>>> queue1, queue2; // And we swap between those to avoid unneseecary copying
            queue1.push_back(v);

            // Split by X
            for (Voxel<R3xS1<FT>>& voxel : queue1) {
                FT left = v.bottomLeft().getX(); 
                FT right = v.topRight().getX();
                for (int i = 0; i < numSplitX; i++) {
                    FT a = left + ((double)i / (double)numSplitX) * (right - left);
                    FT b = left + ((double)(i+1) / (double)numSplitX) * (right - left);
                    queue2.push_back(Voxel(
                        R3xS1(a, voxel.bottomLeft().getY(), voxel.bottomLeft().getZ(), voxel.bottomLeft().getR()),
                        R3xS1(b, voxel.topRight().getY(), voxel.topRight().getZ(), voxel.topRight().getR())
                    ));
                }
            }

            // Split by Y
            queue1.clear();
            for (Voxel<R3xS1<FT>>& voxel: queue2) {
                FT left = v.bottomLeft().getY();
                FT right = v.topRight().getY();
                for (int i = 0; i < numSplitY; i++) {
                    FT a = left + ((double)i / (double)numSplitY) * (right - left);
                    FT b = left + ((double)(i+1) / (double)numSplitY) * (right - left);
                    queue1.push_back(Voxel(
                        R3xS1(voxel.bottomLeft().getX(), a, voxel.bottomLeft().getZ(), voxel.bottomLeft().getR()),
                        R3xS1(voxel.topRight().getX(), b, voxel.topRight().getZ(), voxel.topRight().getR())
                    ));
                }
            }

            // Split by Z
            queue2.clear();
            for (Voxel<R3xS1<FT>>& voxel: queue1) {
                FT left = v.bottomLeft().getZ();
                FT right = v.topRight().getZ();
                for (int i = 0; i < numSplitZ; i++) {
                    FT a = left + ((double)i / (double)numSplitZ) * (right - left);
                    FT b = left + ((double)(i+1) / (double)numSplitZ) * (right - left);
                    queue2.push_back(Voxel(
                        R3xS1(voxel.bottomLeft().getX(), voxel.bottomLeft().getY(), a, voxel.bottomLeft().getR()),
                        R3xS1(voxel.topRight().getX(), voxel.topRight().getY(), b, voxel.topRight().getR())
                    ));
                }
            }

            // Split by R
            // Notice that since this is last dimension, we append directly into output
            for (Voxel<R3xS1<FT>>& voxel: queue2) {
                FT left = v.bottomLeft().getR();
                FT right = v.topRight().getR();
                for (int i = 0; i < numSplitR; i++) {
                    FT a = left + ((double)i / (double)numSplitR) * (right - left);
                    FT b = left + ((double)(i+1) / (double)numSplitR) * (right - left);
                    out.push_back(Voxel(
                        R3xS1(voxel.bottomLeft().getX(), voxel.bottomLeft().getY(), voxel.bottomLeft().getZ(), a),
                        R3xS1(voxel.topRight().getX(), voxel.topRight().getY(), voxel.topRight().getZ(), b)
                    ));
                }
            }
        }

    protected:
        int numSplitX, numSplitY, numSplitZ, numSplitR;
        int incrementCount;
    };

    template<typename FT>
    class ScheduledSplitter_R3xS1 : public Splitter_R3xS1<FT> {
    public:
        ScheduledSplitter_R3xS1(std::vector<std::vector<int>> schedule) : schedule(schedule), Splitter_R3xS1<FT>() {
            this->incrementCount = -1;
            inc(); // Initialize the 0-level split params
        }

        void inc() override {
            Splitter_R3xS1<FT>::inc();
            this->numSplitX = schedule[this->incrementCount][0];
            this->numSplitY = schedule[this->incrementCount][1];
            this->numSplitZ = schedule[this->incrementCount][2];
            this->numSplitR = schedule[this->incrementCount][3];
            fmt::print("Increment to {} [{},{},{},{}]\n", this->incrementCount, this->numSplitX, this->numSplitY, this->numSplitZ, this->numSplitR);
        }

    protected:
        std::vector<std::vector<int>> schedule;
    };

}

#endif