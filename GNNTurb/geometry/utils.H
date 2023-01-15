#pragma once

namespace geometry {

struct FaceIdInfo {
    const int64_t faceid; // face id
    const bool is_internal; // is internal face
    const int64_t patchi; // patch id
    const int64_t pFacei; // face id in the patch

    FaceIdInfo(): faceid(0), is_internal(true), patchi(-1), pFacei(-1) {}

    FaceIdInfo(const int64_t faceid): 
        faceid(faceid), is_internal(true), patchi(-1), pFacei(-1) {}

    FaceIdInfo(const int64_t faceid, const int64_t patchi, const int64_t pFacei): 
        faceid(faceid), is_internal(false), patchi(patchi), pFacei(pFacei) {}
};


struct Pos { // for test
    float x_;
    float y_;
    float z_;
    Pos(float x, float y, float z): x_(x), y_(y), z_(z) {}
    float x() {return x_; }
    float y() {return y_; }
    float z() {return z_; }
};

} // namespace geometry
