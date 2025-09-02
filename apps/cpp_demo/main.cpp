#include <format>
#include <iostream>

#include <sdsl/bindings/sdsl_binding.hpp>
#include <sdsl/bindings/sdsl_cgal.hpp>

#include <sdsl/sdsl.hpp>
#include <sdsl/actions/action_R3xS2.hpp>
#include <sdsl/environments/env_R3_pcd.hpp>
#include <sdsl/splitters/splitter_R3xS1.hpp>
#include <sdsl/configurations/config_R3xS1.hpp>
#include <sdsl/predicates/predicate_static.hpp>
#include <sdsl/predicates/predicate_dynamic.hpp>
using namespace sdsl;

using Point_3 = Kernel::Point_3;

std::vector<Point_3> loadFl4Environment();
std::vector<R3xS2<FT>> loadOdometry();
std::vector<FT> loadMeasurements();

std::vector<std::vector<int>> createSchedule() {
    std::vector<std::vector<int>> schedule;
    
    for (int i = 0; i < 10; i++) {
        schedule.push_back(std::vector<int>());
    }

    schedule[0].push_back(16); schedule[0].push_back(16); schedule[0].push_back(1); schedule[0].push_back(32); 
    schedule[1].push_back(2); schedule[1].push_back(2); schedule[1].push_back(1); schedule[1].push_back(2); 

    for (int i = 2; i < schedule.size(); i++) {
        schedule[i].push_back(2); schedule[i].push_back(2); schedule[i].push_back(1); schedule[i].push_back(1); 
    }

    return schedule;
}

int main() {
    std::vector<Point_3> points = loadFl4Environment();
    std::vector<R3xS2<FT>> odometry = loadOdometry();
    std::vector<FT> measurements = loadMeasurements();
    std::vector<std::vector<int>> schedule = createSchedule();
    int k_ = measurements.size() - 1;
    double errorBound = 0.05;
    int recursionDepth = 5;

    Env_R3_PCD<Kernel> env(points);
    ScheduledSplitter_R3xS1<FT> splitter = ScheduledSplitter_R3xS1<FT>(schedule);
    Predicate_Dynamic_Naive_Fast<R3xS1<FT>, R3xS2<FT>, FT, Env_R3_PCD<Kernel>> predicate(measurements.size(), k_);
    
    localize<R3xS1<FT>, ScheduledSplitter_R3xS1<FT>, R3xS2<FT>, FT, Env_R3_PCD<Kernel>>(
        env, odometry, measurements, errorBound, recursionDepth, predicate, splitter
    );


    return 0;
}



std::vector<R3xS2<FT>> loadOdometry() {
    std::vector<R3xS2<FT>> odometry;
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,-0.9998476900853955,-0.01745269695201492,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,-0.9174943859278823,-0.39774872946474404,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,-0.6962258817020573,-0.717822764788212,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,-0.3695438085947545,-0.9292133089497177,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,0.013089835325736229,-0.9999143244354214,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,0.39374158472891396,-0.9192211727626619,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,0.7147780472630443,-0.6993513731671864,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,0.9275920106049149,-0.3735947829693709,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,0.9999619214149178,0.00872672448206043,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,0.9209304585439644,0.38972694354638027,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,0.7024635496803282,0.7117197210774148,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,0.3776386444722767,0.9259530518342393,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,-0.004363447490262324,0.9999904801176858,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,-0.38570488235190326,0.922622210728695,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,-0.708647844458801,0.7055623519887487,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,-0.9242964638418294,0.3816753161123859,0.0));
    odometry.push_back(R3xS2<FT>(0.0,0.0,0.0,-0.9999999999999962,-8.742278000372475e-08,0.0));
    return odometry;
}
std::vector<FT> loadMeasurements() {
    std::vector<FT> measurements;
    measurements.push_back(2.444000005722046);
    measurements.push_back(2.8359999656677246);
    measurements.push_back(6.159999847412109);
    measurements.push_back(1.694000005722046);
    measurements.push_back(1.027999997138977);
    measurements.push_back(2.055999994277954);
    measurements.push_back(2.0299999713897705);
    measurements.push_back(2.888000011444092);
    measurements.push_back(2.0160000324249268);
    measurements.push_back(1.7280000448226929);
    measurements.push_back(1.6519999504089355);
    measurements.push_back(2.240000009536743);
    measurements.push_back(0.9760000109672546);
    measurements.push_back(2.1440000534057617);
    measurements.push_back(0.8119999766349792);
    measurements.push_back(2.240000009536743);
    measurements.push_back(2.4519999027252197);
    return measurements;
}

// Currently, the point cloud is hard-coded in the C++ file, for simplicity
std::vector<Point_3> loadFl4Environment() {
    std::vector<Point_3> result;
    
    // Load from the file "points.txt", which is in the format x,y,z\n
    // Error if file not found
    std::ifstream file("points.txt");
    if (!file) {
        std::cerr << "Error: Could not open file points.txt" << std::endl;
        return result;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double x, y, z;
        char comma;
        if (iss >> x >> comma >> y >> comma >> z) {
            result.push_back(Point_3(x, y, z));
        }
    }

    return result;
}