#ifndef GRAVITY_ANIM_STATE_H
#define GRAVITY_ANIM_STATE_H

#include "geometry.h"
#include <vector>

struct GravityAnimationState {
    int meshIndex;
    Vec3 dir;
    float step;
    int iteration;
    int maxIterations;
    bool active;
    bool finished;

    // Two-phase step: 0 = Move, 1 = Check
    int phase;
    Vec3 lastMove; // To undo if needed
};

#endif
