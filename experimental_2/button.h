#ifndef BUTTON_H
#define BUTTON_H

#include <functional>

struct Button {
    int x, y, w, h;
    const char* label;
    bool hover;
    char shortcut; // Keyboard shortcut
    std::function<void()> action;

    void draw(int windowW, int windowH);
    bool isInside(int mx, int my) const;
    void onClick();
};

#endif
