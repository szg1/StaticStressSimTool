#include "button.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

static void drawString(void *font, const char *string) {
    while (*string) glutBitmapCharacter(font, *string++);
}

void Button::draw(int windowW, int windowH) {
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    // Switch to 2D Orthographic projection
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, windowW, windowH, 0); // Top-left origin
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Draw Button Background
    if (hover) glColor4f(0.3f, 0.3f, 0.3f, 0.9f); // Lighter on hover
    else glColor4f(0.15f, 0.15f, 0.15f, 0.8f); // Dark semi-transparent

    glBegin(GL_QUADS);
        glVertex2i(x, y);
        glVertex2i(x + w, y);
        glVertex2i(x + w, y + h);
        glVertex2i(x, y + h);
    glEnd();

    // Draw Border
    glColor3f(0.5f, 0.5f, 0.5f);
    glLineWidth(1.0f);
    glBegin(GL_LINE_LOOP);
        glVertex2i(x, y);
        glVertex2i(x + w, y);
        glVertex2i(x + w, y + h);
        glVertex2i(x, y + h);
    glEnd();

    // Draw Text
    glColor3f(1.0f, 1.0f, 1.0f);
    // Center text roughly
    int textX = x + 15;
    int textY = y + 25;
    glRasterPos2i(textX, textY);
    drawString(GLUT_BITMAP_HELVETICA_18, label);

    // Restore 3D projection
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
}

bool Button::isInside(int mx, int my) const {
    return (mx >= x && mx <= x + w &&
            my >= y && my <= y + h);
}

void Button::onClick() {
    if (action) {
        action();
    }
}
