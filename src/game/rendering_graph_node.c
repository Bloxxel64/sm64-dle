#include <PR/ultratypes.h>

#include "area.h"
#include "engine/math_util.h"
#include "game_init.h"
#include "gfx_dimensions.h"
#include "main.h"
#include "memory.h"
#include "print.h"
#include "rendering_graph_node.h"
#include "shadow.h"
#include "sm64.h"

#include <point_lights.h>

/**
 * This file contains the code that processes the scene graph for rendering.
 * The scene graph is responsible for drawing everything except the HUD / text boxes.
 * First the root of the scene graph is processed when geo_process_root
 * is called from level_script.c. The rest of the tree is traversed recursively
 * using the function geo_process_node_and_siblings, which switches over all
 * geo node types and calls a specialized function accordingly.
 * The types are defined in engine/graph_node.h
 *
 * The scene graph typically looks like:
 * - Root (viewport)
 *  - Master list
 *   - Ortho projection
 *    - Background (skybox)
 *  - Master list
 *   - Perspective
 *    - Camera
 *     - <area-specific display lists>
 *     - Object parent
 *      - <group with 240 object nodes>
 *  - Master list
 *   - Script node (Cannon overlay)
 *
 */

s16 gMatStackIndex;
Mat4 gMatStack[32];
Mtx *gMatStackFixed[32];
#ifdef HIGH_FPS_PC
Mat4 gMatStackInterpolated[32];
Mtx *gMatStackInterpolatedFixed[32];
#endif

/**
 * Animation nodes have state in global variables, so this struct captures
 * the animation state so a 'context switch' can be made when rendering the
 * held object.
 */
struct GeoAnimState {
    /*0x00*/ u8 type;
    /*0x01*/ u8 enabled;
    /*0x02*/ s16 frame;
    /*0x04*/ f32 translationMultiplier;
    /*0x08*/ u16 *attribute;
    /*0x0C*/ s16 *data;
#ifdef HIGH_FPS_PC
    s16 prevFrame;
#endif
};

// For some reason, this is a GeoAnimState struct, but the current state consists
// of separate global variables. It won't match EU otherwise.
struct GeoAnimState gGeoTempState;

u8 gCurrAnimType;
u8 gCurrAnimEnabled;
s16 gCurrAnimFrame;
#ifdef HIGH_FPS_PC
s16 gPrevAnimFrame;
#endif
f32 gCurrAnimTranslationMultiplier;
u16 *gCurrAnimAttribute;
s16 *gCurrAnimData;

struct AllocOnlyPool *gDisplayListHeap;

struct RenderModeContainer {
    u32 modes[8];
};

/* Rendermode settings for cycle 1 for all 8 layers. */
struct RenderModeContainer renderModeTable_1Cycle[2] = { { {
    G_RM_OPA_SURF,
    G_RM_AA_OPA_SURF,
    G_RM_AA_OPA_SURF,
    G_RM_AA_OPA_SURF,
    G_RM_AA_TEX_EDGE,
    G_RM_AA_XLU_SURF,
    G_RM_AA_XLU_SURF,
    G_RM_AA_XLU_SURF,
    } },
    { {
    /* z-buffered */
    G_RM_ZB_OPA_SURF,
    G_RM_AA_ZB_OPA_SURF,
    G_RM_AA_ZB_OPA_DECAL,
    G_RM_AA_ZB_OPA_INTER,
    G_RM_AA_ZB_TEX_EDGE,
    G_RM_AA_ZB_XLU_SURF,
    G_RM_AA_ZB_XLU_DECAL,
    G_RM_AA_ZB_XLU_INTER,
    } } };

/* Rendermode settings for cycle 2 for all 8 layers. */
struct RenderModeContainer renderModeTable_2Cycle[2] = { { {
    G_RM_OPA_SURF2,
    G_RM_AA_OPA_SURF2,
    G_RM_AA_OPA_SURF2,
    G_RM_AA_OPA_SURF2,
    G_RM_AA_TEX_EDGE2,
    G_RM_AA_XLU_SURF2,
    G_RM_AA_XLU_SURF2,
    G_RM_AA_XLU_SURF2,
    } },
    { {
    /* z-buffered */
    G_RM_ZB_OPA_SURF2,
    G_RM_AA_ZB_OPA_SURF2,
    G_RM_AA_ZB_OPA_DECAL2,
    G_RM_AA_ZB_OPA_INTER2,
    G_RM_AA_ZB_TEX_EDGE2,
    G_RM_AA_ZB_XLU_SURF2,
    G_RM_AA_ZB_XLU_DECAL2,
    G_RM_AA_ZB_XLU_INTER2,
    } } };

struct GraphNodeRoot *gCurGraphNodeRoot = NULL;
struct GraphNodeMasterList *gCurGraphNodeMasterList = NULL;
struct GraphNodePerspective *gCurGraphNodeCamFrustum = NULL;
struct GraphNodeCamera *gCurGraphNodeCamera = NULL;
struct GraphNodeObject *gCurGraphNodeObject = NULL;
struct GraphNodeHeldObject *gCurGraphNodeHeldObject = NULL;
u16 gAreaUpdateCounter = 0;

#ifdef F3DEX_GBI_2
LookAt lookAt;
#endif

#ifdef HIGH_FPS_PC
static Gfx *sPerspectivePos;
static Mtx *sPerspectiveMtx;

struct {
    Gfx *pos;
    void *mtx;
    void *displayList;
} gMtxTbl[6400];
s32 gMtxTblSize;

static Gfx *sViewportPos;
static Vp sPrevViewport;

void mtx_patch_interpolated(void) {
    s32 i;

    if (sPerspectivePos != NULL) {
        gSPMatrix(sPerspectivePos, VIRTUAL_TO_PHYSICAL(sPerspectiveMtx), G_MTX_PROJECTION | G_MTX_LOAD | G_MTX_NOPUSH);
    }

    for (i = 0; i < gMtxTblSize; i++) {
        Gfx *pos = gMtxTbl[i].pos;
        gSPMatrix(pos++, VIRTUAL_TO_PHYSICAL(gMtxTbl[i].mtx),
                  G_MTX_MODELVIEW | G_MTX_LOAD | G_MTX_NOPUSH);
        gSPDisplayList(pos++, gMtxTbl[i].displayList);
    }

    if (sViewportPos != NULL) {
        Gfx *saved = gDisplayListHead;
        gDisplayListHead = sViewportPos;
        make_viewport_clip_rect(&sPrevViewport);
        gSPViewport(gDisplayListHead, VIRTUAL_TO_PHYSICAL(&sPrevViewport));
        gDisplayListHead = saved;
    }

    gMtxTblSize = 0;
    sPerspectivePos = NULL;
    sViewportPos = NULL;
}
#endif

/**
 * Process a master list node.
 */
static void geo_process_master_list_sub(struct GraphNodeMasterList *node) {
    struct DisplayListNode *currList;
    s32 i;
    s32 enableZBuffer = (node->node.flags & GRAPH_RENDER_Z_BUFFER) != 0;
    struct RenderModeContainer *modeList = &renderModeTable_1Cycle[enableZBuffer];
    struct RenderModeContainer *mode2List = &renderModeTable_2Cycle[enableZBuffer];

    // @bug This is where the LookAt values should be calculated but aren't.
    // As a result, environment mapping is broken on Fast3DEX2 without the
    // changes below.
#ifdef F3DEX_GBI_2
    Mtx lMtx;
    guLookAtReflect(&lMtx, &lookAt, 0, 0, 0, /* eye */ 0, 0, 1, /* at */ 1, 0, 0 /* up */);
#endif

    if (enableZBuffer != 0) {
        gDPPipeSync(gDisplayListHead++);
        gSPSetGeometryMode(gDisplayListHead++, G_ZBUFFER);
    }

    for (i = 0; i < GFX_NUM_MASTER_LISTS; i++) {
        if ((currList = node->listHeads[i]) != NULL) {
            gDPSetRenderMode(gDisplayListHead++, modeList->modes[i], mode2List->modes[i]);
            while (currList != NULL) {
#ifdef HIGH_FPS_PC
                if ((u32) gMtxTblSize < sizeof(gMtxTbl) / sizeof(gMtxTbl[0])) {
                    gMtxTbl[gMtxTblSize].pos = gDisplayListHead;
                    gMtxTbl[gMtxTblSize].mtx = currList->transform;
                    gMtxTbl[gMtxTblSize++].displayList = currList->displayList;
                }
                gSPMatrix(gDisplayListHead++, VIRTUAL_TO_PHYSICAL(currList->transformInterpolated),
                          G_MTX_MODELVIEW | G_MTX_LOAD | G_MTX_NOPUSH);
                gSPDisplayList(gDisplayListHead++, currList->displayListInterpolated);
#else
                gSPMatrix(gDisplayListHead++, VIRTUAL_TO_PHYSICAL(currList->transform),
                          G_MTX_MODELVIEW | G_MTX_LOAD | G_MTX_NOPUSH);
                gSPDisplayList(gDisplayListHead++, currList->displayList);
#endif
                currList = currList->next;
            }
        }
    }
    if (enableZBuffer != 0) {
        gDPPipeSync(gDisplayListHead++);
        gSPClearGeometryMode(gDisplayListHead++, G_ZBUFFER);
    }
}

/**
 * Appends the display list to one of the master lists based on the layer
 * parameter. Look at the RenderModeContainer struct to see the corresponding
 * render modes of layers.
 */
#ifdef HIGH_FPS_PC
static void geo_append_display_list2(void *displayList, void *displayListInterpolated, s16 layer)
#else
static void geo_append_display_list(void *displayList, s16 layer)
#endif
{

#ifdef F3DEX_GBI_2
    gSPLookAt(gDisplayListHead++, &lookAt);
#endif
    if (gCurGraphNodeMasterList != 0) {
        struct DisplayListNode *listNode =
            alloc_only_pool_alloc(gDisplayListHeap, sizeof(struct DisplayListNode));

        listNode->transform = gMatStackFixed[gMatStackIndex];
        listNode->displayList = displayList;
#ifdef HIGH_FPS_PC
        listNode->transformInterpolated = gMatStackInterpolatedFixed[gMatStackIndex];
        listNode->displayListInterpolated = displayListInterpolated;
#endif
        listNode->next = 0;
        if (gCurGraphNodeMasterList->listHeads[layer] == 0) {
            gCurGraphNodeMasterList->listHeads[layer] = listNode;
        } else {
            gCurGraphNodeMasterList->listTails[layer]->next = listNode;
        }
        gCurGraphNodeMasterList->listTails[layer] = listNode;
    }
}

#ifdef HIGH_FPS_PC
static void geo_append_display_list(void *displayList, s16 layer) {
    geo_append_display_list2(displayList, displayList, layer);
}
#endif

/**
 * Process the master list node.
 */
static void geo_process_master_list(struct GraphNodeMasterList *node) {
    s32 i;
    UNUSED u8 filler[4];

    if (gCurGraphNodeMasterList == NULL && node->node.children != NULL) {
        gCurGraphNodeMasterList = node;
        for (i = 0; i < GFX_NUM_MASTER_LISTS; i++) {
            node->listHeads[i] = NULL;
        }
        geo_process_node_and_siblings(node->node.children);
        geo_process_master_list_sub(node);
        gCurGraphNodeMasterList = NULL;
    }
}

/**
 * Process an orthographic projection node.
 */
static void geo_process_ortho_projection(struct GraphNodeOrthoProjection *node) {
    if (node->node.children != NULL) {
        Mtx *mtx = alloc_display_list(sizeof(*mtx));
        f32 left = (gCurGraphNodeRoot->x - gCurGraphNodeRoot->width) / 2.0f * node->scale;
        f32 right = (gCurGraphNodeRoot->x + gCurGraphNodeRoot->width) / 2.0f * node->scale;
        f32 top = (gCurGraphNodeRoot->y - gCurGraphNodeRoot->height) / 2.0f * node->scale;
        f32 bottom = (gCurGraphNodeRoot->y + gCurGraphNodeRoot->height) / 2.0f * node->scale;

        guOrtho(mtx, left, right, bottom, top, -2.0f, 2.0f, 1.0f);
        gSPPerspNormalize(gDisplayListHead++, 0xFFFF);
        gSPMatrix(gDisplayListHead++, VIRTUAL_TO_PHYSICAL(mtx), G_MTX_PROJECTION | G_MTX_LOAD | G_MTX_NOPUSH);

        geo_process_node_and_siblings(node->node.children);
    }
}

/**
 * Process a perspective projection node.
 */
static void geo_process_perspective(struct GraphNodePerspective *node) {
    if (node->fnNode.func != NULL) {
        node->fnNode.func(GEO_CONTEXT_RENDER, &node->fnNode.node, gMatStack[gMatStackIndex]);
    }
    if (node->fnNode.node.children != NULL) {
        u16 perspNorm;
        Mtx *mtx = alloc_display_list(sizeof(*mtx));
#ifdef HIGH_FPS_PC
        Mtx *mtxInterpolated = alloc_display_list(sizeof(*mtxInterpolated));
        f32 fovInterpolated;
#endif

#ifdef VERSION_EU
        f32 aspect = ((f32) gCurGraphNodeRoot->width / (f32) gCurGraphNodeRoot->height) * 1.1f;
#else
        f32 aspect = (f32) gCurGraphNodeRoot->width / (f32) gCurGraphNodeRoot->height;
#endif

        guPerspective(mtx, &perspNorm, node->fov, aspect, node->near, node->far, 1.0f);

#ifdef HIGH_FPS_PC
        if (gGlobalTimer == node->prevTimestamp + 1 && gGlobalTimer != gLakituState.skipCameraInterpolationTimestamp) {

            fovInterpolated = (node->prevFov + node->fov) / 2.0f;
            guPerspective(mtxInterpolated, &perspNorm, fovInterpolated, aspect, node->near, node->far, 1.0f);
            gSPPerspNormalize(gDisplayListHead++, perspNorm);

            sPerspectivePos = gDisplayListHead;
            sPerspectiveMtx = mtx;
            gSPMatrix(gDisplayListHead++, VIRTUAL_TO_PHYSICAL(mtxInterpolated),
                      G_MTX_PROJECTION | G_MTX_LOAD | G_MTX_NOPUSH);
        } else {
            gSPPerspNormalize(gDisplayListHead++, perspNorm);
            gSPMatrix(gDisplayListHead++, VIRTUAL_TO_PHYSICAL(mtx), G_MTX_PROJECTION | G_MTX_LOAD | G_MTX_NOPUSH);
        }
        node->prevFov = node->fov;
        node->prevTimestamp = gGlobalTimer;
#else
        gSPPerspNormalize(gDisplayListHead++, perspNorm);
        gSPMatrix(gDisplayListHead++, VIRTUAL_TO_PHYSICAL(mtx), G_MTX_PROJECTION | G_MTX_LOAD | G_MTX_NOPUSH);
#endif

        gCurGraphNodeCamFrustum = node;
        geo_process_node_and_siblings(node->fnNode.node.children);
        gCurGraphNodeCamFrustum = NULL;
    }
}

/**
 * Process a level of detail node. From the current transformation matrix,
 * the perpendicular distance to the camera is extracted and the children
 * of this node are only processed if that distance is within the render
 * range of this node.
 */
static void geo_process_level_of_detail(struct GraphNodeLevelOfDetail *node) {
#ifdef GBI_FLOATS
    Mtx *mtx = gMatStackFixed[gMatStackIndex];
    s16 distanceFromCam = (s32) -mtx->m[3][2]; // z-component of the translation column
#else
    // The fixed point Mtx type is defined as 16 longs, but it's actually 16
    // shorts for the integer parts followed by 16 shorts for the fraction parts
    Mtx *mtx = gMatStackFixed[gMatStackIndex];
    s16 distanceFromCam = -GET_HIGH_S16_OF_32(mtx->m[1][3]); // z-component of the translation column
#endif

#ifndef TARGET_N64
    // We assume modern hardware is powerful enough to draw the most detailed variant
    distanceFromCam = 0;
#endif

    if (node->minDistance <= distanceFromCam && distanceFromCam < node->maxDistance) {
        if (node->node.children != 0) {
            geo_process_node_and_siblings(node->node.children);
        }
    }
}

/**
 * Process a switch case node. The node's selection function is called
 * if it is 0, and among the node's children, only the selected child is
 * processed next.
 */
static void geo_process_switch(struct GraphNodeSwitchCase *node) {
    struct GraphNode *selectedChild = node->fnNode.node.children;
    s32 i;

    if (node->fnNode.func != NULL) {
        node->fnNode.func(GEO_CONTEXT_RENDER, &node->fnNode.node, gMatStack[gMatStackIndex]);
    }
    for (i = 0; selectedChild != NULL && node->selectedCase > i; i++) {
        selectedChild = selectedChild->next;
    }
    if (selectedChild != NULL) {
        geo_process_node_and_siblings(selectedChild);
    }
}

#ifdef HIGH_FPS_PC
void interpolate_vectors(Vec3f res, Vec3f a, Vec3f b) {
    res[0] = (a[0] + b[0]) / 2.0f;
    res[1] = (a[1] + b[1]) / 2.0f;
    res[2] = (a[2] + b[2]) / 2.0f;
}

void interpolate_vectors_s16(Vec3s res, Vec3s a, Vec3s b) {
    res[0] = (a[0] + b[0]) / 2;
    res[1] = (a[1] + b[1]) / 2;
    res[2] = (a[2] + b[2]) / 2;
}

static s16 interpolate_angle(s16 a, s16 b) {
    s32 absDiff = b - a;
    if (absDiff < 0) {
        absDiff = -absDiff;
    }
    if (absDiff >= 0x4000 && absDiff <= 0xC000) {
        return b;
    }
    if (absDiff <= 0x8000) {
        return (a + b) / 2;
    } else {
        return (a + b) / 2 + 0x8000;
    }
}

static void interpolate_angles(Vec3s res, Vec3s a, Vec3s b) {
    res[0] = interpolate_angle(a[0], b[0]);
    res[1] = interpolate_angle(a[1], b[1]);
    res[2] = interpolate_angle(a[2], b[2]);
}
#endif

extern struct MarioState *gMarioState;

struct SceneLight gPointLights[MAX_POINT_LIGHTS];
s8 gLightDir[3] = {0x28, 0x28, 0x28};
u8 gLightDirTransformEnabled = 0;
u8 gOverrideDirectionalLight = FALSE;
u8 gOverrideAmbientLight = FALSE;
u8 gPointLightCount = 0;
u8 gAreaPointLightCount = 0;

Lights1 gDirectionalLight = gdSPDefLights1(
	0x7F, 0x7F, 0x7F,
	0xFF, 0xFF, 0xFF, 0x28, 0x28, 0x28);

/**
 * Gets the square of the distance between two vectors
 */
f32 vec3f_dist_sq(Vec3f p1, Vec3f p2)
{
    f32 dx = p2[0] - p1[0];
    f32 dy = p2[1] - p1[1];
    f32 dz = p2[2] - p1[2];
    return (dx * dx) + (dy * dy) + (dz * dz);
}

#define MAX_POINT_LIGHT_DIST (3000)

#include "src/engine/surface_collision.h"

int gPointLightCompatibilityMode = 0;
Mat4 *viewMat;

/**
 * Creates a displaylist to set the active point lights closest to a given location
 */
Gfx* createPointLightsDl(Vec3f pos, f32 yOffset)
{
    Gfx *pointLightsDl, *pointLightsDlHead;

    // The lights to be used for this object
    struct SceneLight *lights[MAX_POINT_LIGHTS_ACTIVE];

    // The number of lights selected to be active for this object
    s32 numLightsPicked = 0;

    // The square of the distances to each point light
    f32 distancesSq[MAX_POINT_LIGHTS_ACTIVE];

    // The distance of the furthest light selected
    f32 maxDistanceSq;

    // The index of the furthest away light from this object
    // i.e. if index 1 in lights is the furthest light from this object, then this is 1
    u32 maxIndex = 0;

    // The square of the distance to the current light being checked
    f32 curDistSq;

    // The index of a point light being added (its position in Light *lights[])
    s32 newIndex;

    // Iterator variables
    s32 i,j;
    
    // Raycast variables
    Vec3f dir, hit;
    struct Surface* surf;

    if (gPointLightCount)
    {
        // Probe higher by the given offset (used since most objects have their origin at the bottom)
        pos[1] += yOffset;
    }

    // Find the closest lights
    for (i = 0; i < gPointLightCount; i++)
    {
        // Reset newIndex so we can use it to check if the current light was added
        newIndex = -1;

        // Get the distance to the current light from the object
        curDistSq = vec3f_dist_sq(pos, gPointLights[i].worldPos);

        // Skip this point light if it is too far away to matter
        if (curDistSq > MAX_POINT_LIGHT_DIST * MAX_POINT_LIGHT_DIST) continue;

        // Skip this point light if it is set to occlude and is occluded
        // If the object and the light are at the same position, skip the raycast
        if (curDistSq > 0.01 && gPointLights[i].flags & LIGHT_FLAG_OCCLUDE)
        {
            dir[0] = gPointLights[i].worldPos[0] - pos[0];
            dir[1] = gPointLights[i].worldPos[1] - pos[1];
            dir[2] = gPointLights[i].worldPos[2] - pos[2];
            // Raycast will return a nonnegative distance in the case of a hit
            if (raycast(pos, dir, sqrtf(curDistSq), hit, &surf) >= 0)
            {
                continue;
            }
        }

        // If we haven't filled all the active light slots, just add this one
        if (numLightsPicked < MAX_POINT_LIGHTS_ACTIVE)
        {
            // Record the index this light was placed into and update the picked light count
            newIndex = numLightsPicked;
            numLightsPicked++;
        }
        // Otherwise, we need to check if this one is closer than any of the ones picked thus far
        else
        {
            // Check if this light is closer than the furthest away one to be used
            // If it is, then we remove the furthest light and add this one
            if (curDistSq < distancesSq[maxIndex])
            {
                newIndex = maxIndex;
            }
        }

        // If this light was added, add it to the lights array, update its distance,
        // and update the distance order of the other lights
        if (newIndex != -1)
        {
            // Place this light in the lights array
            lights[newIndex] = &gPointLights[i];

            // Update the distance to this light
            distancesSq[newIndex] = curDistSq;

            // Set this light to the be furthest one away
            // This will get updated in the following loop
            maxDistanceSq = curDistSq;
            maxIndex = newIndex;

            // Iterate over every light, checking to see if it is further than the current furthest
            // If it is, then set it to be the furthest light instead
            for (j = 0; j < MAX_POINT_LIGHTS_ACTIVE; j++)
            {
                // Skip checking the current light, since we assumed it was the furthest already
                if (j == newIndex) continue;

                // Check if the light being checked is further than current furthest
                if (maxDistanceSq < distancesSq[j])
                {
                    maxDistanceSq = distancesSq[j];
                    maxIndex = j;
                }
            }
        }
    }
    
    // Allocate a displaylist with room for each gSPLight and the gSPEndDisplayList
    pointLightsDlHead = pointLightsDl = alloc_display_list(sizeof(Gfx) * (numLightsPicked + 4));

    gSPNumLights(pointLightsDl++, NUMLIGHTS_1 + numLightsPicked);
    
    gSPLight(pointLightsDl++, &gDirectionalLight.l, LIGHT_1);

    // Add the gSPLights to the display list
    for (i = 0; i < numLightsPicked; i++)
    {
        if (gPointLightCompatibilityMode)
        {
            Light *curLight = alloc_display_list(sizeof(Light));
            u8 color[3];
            f32 lightDist;
            f32 lightScale;

            bzero(curLight, sizeof(Light));

            color[0] = lights[i]->l.pl.col[0];
            color[1] = lights[i]->l.pl.col[1];
            color[2] = lights[i]->l.pl.col[2];

            dir[0] = lights[i]->worldPos[0] - pos[0];
            dir[1] = lights[i]->worldPos[1] - pos[1];
            dir[2] = lights[i]->worldPos[2] - pos[2];

            lightDist = sqrtf(distancesSq[i]);
            lightScale = 1.0f / ((1.0f / 65536.0f) * (
                0.25f * lights[i]->l.pl.constant_attenuation +
                2.0f * lightDist * lights[i]->l.pl.linear_attenuation +
                0.3f * lightDist * lightDist * lights[i]->l.pl.quadratic_attenuation) + 1.0f);

            curLight->l.col[0] = curLight->l.colc[0] = (u8)(color[0] * lightScale + 0.5f);
            curLight->l.col[1] = curLight->l.colc[1] = (u8)(color[1] * lightScale + 0.5f);
            curLight->l.col[2] = curLight->l.colc[2] = (u8)(color[2] * lightScale + 0.5f);
            
            dir[0] *= 120.0f / (lightDist);
            dir[1] *= 120.0f / (lightDist);
            dir[2] *= 120.0f / (lightDist);

            curLight->l.dir[0] = (s8)(dir[0] * (*viewMat)[0][0] + dir[1] * (*viewMat)[1][0] + dir[2] * (*viewMat)[2][0]);
            curLight->l.dir[1] = (s8)(dir[0] * (*viewMat)[0][1] + dir[1] * (*viewMat)[1][1] + dir[2] * (*viewMat)[2][1]);
            curLight->l.dir[2] = (s8)(dir[0] * (*viewMat)[0][2] + dir[1] * (*viewMat)[1][2] + dir[2] * (*viewMat)[2][2]);

            gSPLight(pointLightsDl++, curLight, LIGHT_2 + i);
        }
        else
        {
            gSPLight(pointLightsDl++, &lights[i]->l, LIGHT_2 + i);
        }
    }

    if (gPointLightCount)
    {
        // Restore the original position
        pos[1] -= yOffset;
    }

    gSPLight(pointLightsDl++, &gDirectionalLight.a, LIGHT_2 + numLightsPicked);

    // Terminate the display list
    gSPEndDisplayList(pointLightsDl);

    // Return the head of the created display list
    return pointLightsDlHead;
}

// Sets the scene's directional light, overrides whatever may be set in the area's geolayout
void set_directional_light(Vec3f direction, s32 red, s32 green, s32 blue)
{
    Vec3f directionNormalized;
    vec3f_copy(directionNormalized, direction);
    vec3f_normalize(directionNormalized);
    gLightDir[0] = (s8)(s32)(directionNormalized[0] * 0x40);
    gLightDir[1] = (s8)(s32)(directionNormalized[1] * 0x40);
    gLightDir[2] = (s8)(s32)(directionNormalized[2] * 0x40);
    gDirectionalLight.l[0].l.colc[0] = gDirectionalLight.l[0].l.col[0] = red;
    gDirectionalLight.l[0].l.colc[1] = gDirectionalLight.l[0].l.col[1] = green;
    gDirectionalLight.l[0].l.colc[2] = gDirectionalLight.l[0].l.col[2] = blue;
    gLightDirTransformEnabled = TRUE;
    gOverrideDirectionalLight = TRUE;
}

// Sets the scene's ambient light, overrides whatever may be set in the area's geolayout
void set_ambient_light(s32 red, s32 green, s32 blue)
{
    gDirectionalLight.a.l.colc[0] = gDirectionalLight.a.l.col[0] = red;
    gDirectionalLight.a.l.colc[1] = gDirectionalLight.a.l.col[1] = green;
    gDirectionalLight.a.l.colc[2] = gDirectionalLight.a.l.col[2] = blue;
    gOverrideAmbientLight = TRUE;
}

// Emits a point light with the given parameters
void emit_light(Vec3f pos, s32 red, s32 green, s32 blue, u32 constantFalloff, u32 linearFalloff, u32 quadraticFalloff)
{
    gPointLights[gPointLightCount].l.pl.colc[0] = gPointLights[gPointLightCount].l.pl.col[0] = red;
    gPointLights[gPointLightCount].l.pl.colc[1] = gPointLights[gPointLightCount].l.pl.col[1] = green;
    gPointLights[gPointLightCount].l.pl.colc[2] = gPointLights[gPointLightCount].l.pl.col[2] = blue;
    gPointLights[gPointLightCount].l.pl.constant_attenuation = (constantFalloff == 0) ? 1 : constantFalloff;
    gPointLights[gPointLightCount].l.pl.linear_attenuation = linearFalloff;
    gPointLights[gPointLightCount].l.pl.quadratic_attenuation = quadraticFalloff;
    gPointLights[gPointLightCount].worldPos[0] = pos[0];
    gPointLights[gPointLightCount].worldPos[1] = pos[1];
    gPointLights[gPointLightCount].worldPos[2] = pos[2];
    gPointLightCount++;
}

#include <stdio.h>

extern void linear_mtxf_mul_vec3f(Mat4, Vec3f, Vec3f);

/**
 * Process a camera node.
 */
static void geo_process_camera(struct GraphNodeCamera *node) {
    Mat4 cameraTransform;
    Mtx *rollMtx = alloc_display_list(sizeof(*rollMtx));
    Mtx *mtx = alloc_display_list(sizeof(*mtx));
    Gfx *setLightsDL = alloc_display_list(sizeof(Gfx) * 3);
    Gfx *levelLightsDL;
    Vec3f probePos = {0, 0, 0};
    s32 i;
#ifdef HIGH_FPS_PC
    Mtx *mtxInterpolated = alloc_display_list(sizeof(*mtxInterpolated));
    Vec3f posInterpolated;
    Vec3f focusInterpolated;
#endif

    if (node->fnNode.func != NULL) {
        node->fnNode.func(GEO_CONTEXT_RENDER, &node->fnNode.node, gMatStack[gMatStackIndex]);
    }
    mtxf_rotate_xy(rollMtx, node->rollScreen);

    gSPMatrix(gDisplayListHead++, VIRTUAL_TO_PHYSICAL(rollMtx), G_MTX_PROJECTION | G_MTX_MUL | G_MTX_NOPUSH);
    geo_append_display_list(setLightsDL, LAYER_OPAQUE);

    mtxf_lookat(cameraTransform, node->pos, node->focus, node->roll);
    mtxf_mul(gMatStack[gMatStackIndex + 1], cameraTransform, gMatStack[gMatStackIndex]);

#ifdef HIGH_FPS_PC
    if (gGlobalTimer == node->prevTimestamp + 1 && gGlobalTimer != gLakituState.skipCameraInterpolationTimestamp) {
        interpolate_vectors(posInterpolated, node->prevPos, node->pos);
        interpolate_vectors(focusInterpolated, node->prevFocus, node->focus);
        float magnitude = 0;
        int i;
        for (i = 0; i < 3; i++) {
            float diff = node->pos[i] - node->prevPos[i];
            magnitude += diff * diff;
        }
        if (magnitude > 500000) {
            // Observed ~479000 in BBH when toggling R camera
            // Can get over 3 million in VCUTM though...
            vec3f_copy(posInterpolated, node->pos);
            vec3f_copy(focusInterpolated, node->focus);
        }
    } else {
        vec3f_copy(posInterpolated, node->pos);
        vec3f_copy(focusInterpolated, node->focus);
    }
    vec3f_copy(node->prevPos, node->pos);
    vec3f_copy(node->prevFocus, node->focus);
    node->prevTimestamp = gGlobalTimer;
    mtxf_lookat(cameraTransform, posInterpolated, focusInterpolated, node->roll);
    mtxf_mul(gMatStackInterpolated[gMatStackIndex + 1], cameraTransform, gMatStackInterpolated[gMatStackIndex]);
#endif

    gMatStackIndex++;
    mtxf_to_mtx(mtx, gMatStack[gMatStackIndex]);
    

    if (gPlayer1Controller->buttonPressed & L_TRIG)
        gPointLightCompatibilityMode ^= 1;

    viewMat = &gMatStack[gMatStackIndex];

    gMatStackFixed[gMatStackIndex] = mtx;
#ifdef HIGH_FPS_PC
    mtxf_to_mtx(mtxInterpolated, gMatStackInterpolated[gMatStackIndex]);
    gMatStackInterpolatedFixed[gMatStackIndex] = mtxInterpolated;
#endif
    if (node->fnNode.node.children != 0) {
        gCurGraphNodeCamera = node;
        node->matrixPtr = &gMatStack[gMatStackIndex];
#ifdef HIGH_FPS_PC
        node->matrixPtrInterpolated = &gMatStackInterpolated[gMatStackIndex];
#endif
        geo_process_node_and_siblings(node->fnNode.node.children);
        gCurGraphNodeCamera = NULL;
    }


    // Transform the point light positions into screen space
    for (i = 0; i < gPointLightCount; i++)
    {
        Vec3f transformed;
        mtxf_mul_vec3f(gMatStack[gMatStackIndex], gPointLights[i].worldPos, transformed);
        vec3f_to_vec3s(gPointLights[i].l.pl.pos, transformed);
    }

    // Transform the directional light if enabled
    if (gLightDirTransformEnabled)
    {
        gDirectionalLight.l->l.dir[0] = -(s8)(gLightDir[0] * gMatStack[gMatStackIndex][0][0] + gLightDir[1] * gMatStack[gMatStackIndex][1][0] + gLightDir[2] * gMatStack[gMatStackIndex][2][0]);
        gDirectionalLight.l->l.dir[1] = -(s8)(gLightDir[0] * gMatStack[gMatStackIndex][0][1] + gLightDir[1] * gMatStack[gMatStackIndex][1][1] + gLightDir[2] * gMatStack[gMatStackIndex][2][1]);
        gDirectionalLight.l->l.dir[2] = -(s8)(gLightDir[0] * gMatStack[gMatStackIndex][0][2] + gLightDir[1] * gMatStack[gMatStackIndex][1][2] + gLightDir[2] * gMatStack[gMatStackIndex][2][2]);
    }
    else
    {
        gDirectionalLight.l->l.dir[0] = gLightDir[0];
        gDirectionalLight.l->l.dir[1] = gLightDir[1];
        gDirectionalLight.l->l.dir[2] = gLightDir[2];
    }
    gOverrideAmbientLight = FALSE;
    gOverrideDirectionalLight = FALSE;
    
    // Set up the light display list
    // This has to be done after the area's GeoLayout is processed, as
    // some point lights may be defined there instead of by objects
    if (gPointLightCount > 0)
    {
        // Enable point lighting
        gSPSetGeometryMode(setLightsDL++, G_POINT_LIGHTING);
    }
    else
    {
        // Disable point lighting (may not be required, but doesn't hurt)
        gSPClearGeometryMode(setLightsDL++, G_POINT_LIGHTING);
    }

    // Enable the lights closes to the given probe position as the level's lighting
    levelLightsDL = createPointLightsDl(probePos, 300.0f);
    gSPDisplayList(setLightsDL++, levelLightsDL);

    // Terminate the point lighting DL
    gSPEndDisplayList(setLightsDL++);

    gMatStackIndex--;
}

/**
 * Process a translation / rotation node. A transformation matrix based
 * on the node's translation and rotation is created and pushed on both
 * the float and fixed point matrix stacks.
 * For the rest it acts as a normal display list node.
 */
static void geo_process_translation_rotation(struct GraphNodeTranslationRotation *node) {
    Mat4 mtxf;
    Vec3f translation;
    Mtx *mtx = alloc_display_list(sizeof(*mtx));
#ifdef HIGH_FPS_PC
    Mtx *mtxInterpolated = alloc_display_list(sizeof(*mtxInterpolated));
#endif

    vec3s_to_vec3f(translation, node->translation);
    mtxf_rotate_zxy_and_translate(mtxf, translation, node->rotation);
    mtxf_mul(gMatStack[gMatStackIndex + 1], mtxf, gMatStack[gMatStackIndex]);
#ifdef HIGH_FPS_PC
    mtxf_mul(gMatStackInterpolated[gMatStackIndex + 1], mtxf, gMatStackInterpolated[gMatStackIndex]);
#endif
    gMatStackIndex++;
    mtxf_to_mtx(mtx, gMatStack[gMatStackIndex]);
    gMatStackFixed[gMatStackIndex] = mtx;
#ifdef HIGH_FPS_PC
    mtxf_to_mtx(mtxInterpolated, gMatStackInterpolated[gMatStackIndex]);
    gMatStackInterpolatedFixed[gMatStackIndex] = mtxInterpolated;
#endif
    if (node->displayList != NULL) {
        geo_append_display_list(node->displayList, node->node.flags >> 8);
    }
    if (node->node.children != NULL) {
        geo_process_node_and_siblings(node->node.children);
    }
    gMatStackIndex--;
}

/**
 * Process a translation node. A transformation matrix based on the node's
 * translation is created and pushed on both the float and fixed point matrix stacks.
 * For the rest it acts as a normal display list node.
 */
static void geo_process_translation(struct GraphNodeTranslation *node) {
    Mat4 mtxf;
    Vec3f translation;
    Mtx *mtx = alloc_display_list(sizeof(*mtx));
#ifdef HIGH_FPS_PC
    Mtx *mtxInterpolated = alloc_display_list(sizeof(*mtxInterpolated));
#endif

    vec3s_to_vec3f(translation, node->translation);
    mtxf_rotate_zxy_and_translate(mtxf, translation, gVec3sZero);
    mtxf_mul(gMatStack[gMatStackIndex + 1], mtxf, gMatStack[gMatStackIndex]);
#ifdef HIGH_FPS_PC
    mtxf_mul(gMatStackInterpolated[gMatStackIndex + 1], mtxf, gMatStackInterpolated[gMatStackIndex]);
#endif
    gMatStackIndex++;
    mtxf_to_mtx(mtx, gMatStack[gMatStackIndex]);
    gMatStackFixed[gMatStackIndex] = mtx;
#ifdef HIGH_FPS_PC
    mtxf_to_mtx(mtxInterpolated, gMatStackInterpolated[gMatStackIndex]);
    gMatStackInterpolatedFixed[gMatStackIndex] = mtxInterpolated;
#endif
    if (node->displayList != NULL) {
        geo_append_display_list(node->displayList, node->node.flags >> 8);
    }
    if (node->node.children != NULL) {
        geo_process_node_and_siblings(node->node.children);
    }
    gMatStackIndex--;
}

/**
 * Process a rotation node. A transformation matrix based on the node's
 * rotation is created and pushed on both the float and fixed point matrix stacks.
 * For the rest it acts as a normal display list node.
 */
static void geo_process_rotation(struct GraphNodeRotation *node) {
    Mat4 mtxf;
    Mtx *mtx = alloc_display_list(sizeof(*mtx));
#ifdef HIGH_FPS_PC
    Mtx *mtxInterpolated = alloc_display_list(sizeof(*mtxInterpolated));
    Vec3s rotationInterpolated;
#endif

    mtxf_rotate_zxy_and_translate(mtxf, gVec3fZero, node->rotation);
    mtxf_mul(gMatStack[gMatStackIndex + 1], mtxf, gMatStack[gMatStackIndex]);
#ifdef HIGH_FPS_PC
    if (gGlobalTimer == node->prevTimestamp + 1) {
        interpolate_angles(rotationInterpolated, node->prevRotation, node->rotation);
        mtxf_rotate_zxy_and_translate(mtxf, gVec3fZero, rotationInterpolated);
    }
    vec3s_copy(node->prevRotation, node->rotation);
    node->prevTimestamp = gGlobalTimer;
    mtxf_mul(gMatStackInterpolated[gMatStackIndex + 1], mtxf, gMatStackInterpolated[gMatStackIndex]);
#endif
    gMatStackIndex++;
    mtxf_to_mtx(mtx, gMatStack[gMatStackIndex]);
    gMatStackFixed[gMatStackIndex] = mtx;
#ifdef HIGH_FPS_PC
    mtxf_to_mtx(mtxInterpolated, gMatStackInterpolated[gMatStackIndex]);
    gMatStackInterpolatedFixed[gMatStackIndex] = mtxInterpolated;
#endif
    if (node->displayList != NULL) {
        geo_append_display_list(node->displayList, node->node.flags >> 8);
    }
    if (node->node.children != NULL) {
        geo_process_node_and_siblings(node->node.children);
    }
    gMatStackIndex--;
}

/**
 * Process a scaling node. A transformation matrix based on the node's
 * scale is created and pushed on both the float and fixed point matrix stacks.
 * For the rest it acts as a normal display list node.
 */
static void geo_process_scale(struct GraphNodeScale *node) {
    UNUSED Mat4 transform;
    Vec3f scaleVec;
    Mtx *mtx = alloc_display_list(sizeof(*mtx));
#ifdef HIGH_FPS_PC
    Mtx *mtxInterpolated = alloc_display_list(sizeof(*mtxInterpolated));
#endif

    vec3f_set(scaleVec, node->scale, node->scale, node->scale);
    mtxf_scale_vec3f(gMatStack[gMatStackIndex + 1], gMatStack[gMatStackIndex], scaleVec);
#ifdef HIGH_FPS_PC
    mtxf_scale_vec3f(gMatStackInterpolated[gMatStackIndex + 1], gMatStackInterpolated[gMatStackIndex], scaleVec);
#endif
    gMatStackIndex++;
    mtxf_to_mtx(mtx, gMatStack[gMatStackIndex]);
    gMatStackFixed[gMatStackIndex] = mtx;
#ifdef HIGH_FPS_PC
    mtxf_to_mtx(mtxInterpolated, gMatStackInterpolated[gMatStackIndex]);
    gMatStackInterpolatedFixed[gMatStackIndex] = mtxInterpolated;
#endif
    if (node->displayList != NULL) {
        geo_append_display_list(node->displayList, node->node.flags >> 8);
    }
    if (node->node.children != NULL) {
        geo_process_node_and_siblings(node->node.children);
    }
    gMatStackIndex--;
}

/**
 * Process a billboard node. A transformation matrix is created that makes its
 * children face the camera, and it is pushed on the floating point and fixed
 * point matrix stacks.
 * For the rest it acts as a normal display list node.
 */
static void geo_process_billboard(struct GraphNodeBillboard *node) {
    Vec3f translation;
    Mtx *mtx = alloc_display_list(sizeof(*mtx));
#ifdef HIGH_FPS_PC
    Mtx *mtxInterpolated = alloc_display_list(sizeof(*mtxInterpolated));
#endif

    gMatStackIndex++;
    vec3s_to_vec3f(translation, node->translation);
    mtxf_billboard(gMatStack[gMatStackIndex], gMatStack[gMatStackIndex - 1], translation,
                   gCurGraphNodeCamera->roll);
#ifdef HIGH_FPS_PC
    mtxf_billboard(gMatStackInterpolated[gMatStackIndex], gMatStackInterpolated[gMatStackIndex - 1], translation,
                   gCurGraphNodeCamera->roll);
#endif
    if (gCurGraphNodeHeldObject != NULL) {
        mtxf_scale_vec3f(gMatStack[gMatStackIndex], gMatStack[gMatStackIndex],
                         gCurGraphNodeHeldObject->objNode->header.gfx.scale);
#ifdef HIGH_FPS_PC
        mtxf_scale_vec3f(gMatStackInterpolated[gMatStackIndex], gMatStackInterpolated[gMatStackIndex],
                         gCurGraphNodeHeldObject->objNode->header.gfx.scale);
#endif
    } else if (gCurGraphNodeObject != NULL) {
        mtxf_scale_vec3f(gMatStack[gMatStackIndex], gMatStack[gMatStackIndex],
                         gCurGraphNodeObject->scale);
#ifdef HIGH_FPS_PC
        mtxf_scale_vec3f(gMatStackInterpolated[gMatStackIndex], gMatStackInterpolated[gMatStackIndex],
                         gCurGraphNodeObject->scale);
#endif
    }

    mtxf_to_mtx(mtx, gMatStack[gMatStackIndex]);
    gMatStackFixed[gMatStackIndex] = mtx;
#ifdef HIGH_FPS_PC
    mtxf_to_mtx(mtxInterpolated, gMatStackInterpolated[gMatStackIndex]);
    gMatStackInterpolatedFixed[gMatStackIndex] = mtxInterpolated;
#endif
    if (node->displayList != NULL) {
        geo_append_display_list(node->displayList, node->node.flags >> 8);
    }
    if (node->node.children != NULL) {
        geo_process_node_and_siblings(node->node.children);
    }
    gMatStackIndex--;
}

/**
 * Process a display list node. It draws a display list without first pushing
 * a transformation on the stack, so all transformations are inherited from the
 * parent node. It processes its children if it has them.
 */
static void geo_process_display_list(struct GraphNodeDisplayList *node) {
    if (node->displayList != NULL) {
        geo_append_display_list(node->displayList, node->node.flags >> 8);
    }
    if (node->node.children != NULL) {
        geo_process_node_and_siblings(node->node.children);
    }
}

/**
 * Process a generated list. Instead of storing a pointer to a display list,
 * the list is generated on the fly by a function.
 */
static void geo_process_generated_list(struct GraphNodeGenerated *node) {
    if (node->fnNode.func != NULL) {
        Gfx *list = node->fnNode.func(GEO_CONTEXT_RENDER, &node->fnNode.node,
                                     (struct AllocOnlyPool *) gMatStack[gMatStackIndex]);

        if (list != NULL) {
            geo_append_display_list((void *) VIRTUAL_TO_PHYSICAL(list), node->fnNode.node.flags >> 8);
        }
    }
    if (node->fnNode.node.children != NULL) {
        geo_process_node_and_siblings(node->fnNode.node.children);
    }
}

/**
 * Process a background node. Tries to retrieve a background display list from
 * the function of the node. If that function is null or returns null, a black
 * rectangle is drawn instead.
 */
static void geo_process_background(struct GraphNodeBackground *node) {
    Gfx *list = NULL;
#ifdef HIGH_FPS_PC
    Gfx *listInterpolated = NULL;
#endif

    if (node->fnNode.func != NULL) {
#ifdef HIGH_FPS_PC
        Vec3f posCopy;
        Vec3f focusCopy;
        Vec3f posInterpolated;
        Vec3f focusInterpolated;

        if (gGlobalTimer == node->prevCameraTimestamp + 1 &&
            gGlobalTimer != gLakituState.skipCameraInterpolationTimestamp) {
            interpolate_vectors(posInterpolated, node->prevCameraPos, gLakituState.pos);
            interpolate_vectors(focusInterpolated, node->prevCameraFocus, gLakituState.focus);
        } else {
            vec3f_copy(posInterpolated, gLakituState.pos);
            vec3f_copy(focusInterpolated, gLakituState.focus);
        }
        vec3f_copy(node->prevCameraPos, gLakituState.pos);
        vec3f_copy(node->prevCameraFocus, gLakituState.focus);
        node->prevCameraTimestamp = gGlobalTimer;
#endif

        list = node->fnNode.func(GEO_CONTEXT_RENDER, &node->fnNode.node,
                                 (struct AllocOnlyPool *) gMatStack[gMatStackIndex]);
#ifdef HIGH_FPS_PC
        vec3f_copy(posCopy, gLakituState.pos);
        vec3f_copy(focusCopy, gLakituState.focus);
        vec3f_copy(gLakituState.pos, posInterpolated);
        vec3f_copy(gLakituState.focus, focusInterpolated);
        listInterpolated = node->fnNode.func(GEO_CONTEXT_RENDER, &node->fnNode.node, NULL);
        vec3f_copy(gLakituState.pos, posCopy);
        vec3f_copy(gLakituState.focus, focusCopy);
#endif
    }
    if (list != NULL) {
#ifdef HIGH_FPS_PC
        geo_append_display_list2((void *) VIRTUAL_TO_PHYSICAL(list),
                                 (void *) VIRTUAL_TO_PHYSICAL(listInterpolated), node->fnNode.node.flags >> 8);
#else
        geo_append_display_list((void *) VIRTUAL_TO_PHYSICAL(list), node->fnNode.node.flags >> 8);
#endif
    } else if (gCurGraphNodeMasterList != NULL) {
#ifndef F3DEX_GBI_2E
        Gfx *gfxStart = alloc_display_list(sizeof(Gfx) * 7);
#else
        Gfx *gfxStart = alloc_display_list(sizeof(Gfx) * 8);
#endif
        Gfx *gfx = gfxStart;

        gDPPipeSync(gfx++);
        gDPSetCycleType(gfx++, G_CYC_FILL);
        gDPSetFillColor(gfx++, node->background);
#ifdef TARGET_N3DS
        gDPForceFlush(gfx++);
        gDPSet2d(gfx++, 1);
#endif
        gDPFillRectangle(gfx++, GFX_DIMENSIONS_RECT_FROM_LEFT_EDGE(0), BORDER_HEIGHT,
        GFX_DIMENSIONS_RECT_FROM_RIGHT_EDGE(0) - 1, SCREEN_HEIGHT - BORDER_HEIGHT - 1);
#ifdef TARGET_N3DS
        gDPForceFlush(gfx++);
        gDPSet2d(gfx++, 0);
#endif
        gDPPipeSync(gfx++);
        gDPSetCycleType(gfx++, G_CYC_1CYCLE);
        gSPEndDisplayList(gfx++);

        geo_append_display_list((void *) VIRTUAL_TO_PHYSICAL(gfxStart), 0);
    }
    if (node->fnNode.node.children != NULL) {
        geo_process_node_and_siblings(node->fnNode.node.children);
    }
}

static void anim_process(Vec3f translation, Vec3s rotation, u8 *animType, s16 animFrame, u16 **animAttribute) {
    if (*animType == ANIM_TYPE_TRANSLATION) {
        translation[0] += gCurrAnimData[retrieve_animation_index(animFrame, animAttribute)]
                          * gCurrAnimTranslationMultiplier;
        translation[1] += gCurrAnimData[retrieve_animation_index(animFrame, animAttribute)]
                          * gCurrAnimTranslationMultiplier;
        translation[2] += gCurrAnimData[retrieve_animation_index(animFrame, animAttribute)]
                          * gCurrAnimTranslationMultiplier;
        *animType = ANIM_TYPE_ROTATION;
    } else {
        if (*animType == ANIM_TYPE_LATERAL_TRANSLATION) {
            translation[0] +=
                gCurrAnimData[retrieve_animation_index(animFrame, animAttribute)]
                * gCurrAnimTranslationMultiplier;
            *animAttribute += 2;
            translation[2] +=
                gCurrAnimData[retrieve_animation_index(animFrame, animAttribute)]
                * gCurrAnimTranslationMultiplier;
            *animType = ANIM_TYPE_ROTATION;
        } else {
            if (*animType == ANIM_TYPE_VERTICAL_TRANSLATION) {
                *animAttribute += 2;
                translation[1] +=
                    gCurrAnimData[retrieve_animation_index(animFrame, animAttribute)]
                    * gCurrAnimTranslationMultiplier;
                *animAttribute += 2;
                *animType = ANIM_TYPE_ROTATION;
            } else if (*animType == ANIM_TYPE_NO_TRANSLATION) {
                *animAttribute += 6;
                *animType = ANIM_TYPE_ROTATION;
            }
        }
    }

    if (*animType == ANIM_TYPE_ROTATION) {
        rotation[0] = gCurrAnimData[retrieve_animation_index(animFrame, animAttribute)];
        rotation[1] = gCurrAnimData[retrieve_animation_index(animFrame, animAttribute)];
        rotation[2] = gCurrAnimData[retrieve_animation_index(animFrame, animAttribute)];
    }
}

/**
 * Render an animated part. The current animation state is not part of the node
 * but set in global variables. If an animated part is skipped, everything afterwards desyncs.
 */
static void geo_process_animated_part(struct GraphNodeAnimatedPart *node) {
    Mat4 matrix;
    Vec3s rotation;
    Vec3f translation;
#ifdef HIGH_FPS_PC
    Vec3s rotationInterpolated;
    Vec3f translationInterpolated;
#endif
    Mtx *matrixPtr = alloc_display_list(sizeof(*matrixPtr));
#ifdef HIGH_FPS_PC
    Mtx *mtxInterpolated = alloc_display_list(sizeof(*mtxInterpolated));
    u16 *animAttribute = gCurrAnimAttribute;
    u8 animType = gCurrAnimType;
#endif

    vec3s_copy(rotation, gVec3sZero);
    vec3f_set(translation, node->translation[0], node->translation[1], node->translation[2]);
#ifdef HIGH_FPS_PC
    vec3s_copy(rotationInterpolated, rotation);
    vec3f_copy(translationInterpolated, translation);
#endif

#ifdef HIGH_FPS_PC
    anim_process(translationInterpolated, rotationInterpolated, &animType, gPrevAnimFrame, &animAttribute);
#endif
    anim_process(translation, rotation, &gCurrAnimType, gCurrAnimFrame, &gCurrAnimAttribute);
#ifdef HIGH_FPS_PC
    interpolate_vectors(translationInterpolated, translationInterpolated, translation);
    interpolate_angles(rotationInterpolated, rotationInterpolated, rotation);
#endif

    mtxf_rotate_xyz_and_translate(matrix, translation, rotation);
    mtxf_mul(gMatStack[gMatStackIndex + 1], matrix, gMatStack[gMatStackIndex]);
#ifdef HIGH_FPS_PC
    mtxf_rotate_xyz_and_translate(matrix, translationInterpolated, rotationInterpolated);
    mtxf_mul(gMatStackInterpolated[gMatStackIndex + 1], matrix, gMatStackInterpolated[gMatStackIndex]);
#endif
    gMatStackIndex++;
    mtxf_to_mtx(matrixPtr, gMatStack[gMatStackIndex]);
    gMatStackFixed[gMatStackIndex] = matrixPtr;
#ifdef HIGH_FPS_PC
    mtxf_to_mtx(mtxInterpolated, gMatStackInterpolated[gMatStackIndex]);
    gMatStackInterpolatedFixed[gMatStackIndex] = mtxInterpolated;
#endif
    if (node->displayList != NULL) {
        geo_append_display_list(node->displayList, node->node.flags >> 8);
    }
    if (node->node.children != NULL) {
        geo_process_node_and_siblings(node->node.children);
    }
    gMatStackIndex--;
}

/**
 * Initialize the animation-related global variables for the currently drawn
 * object's animation.
 */
void geo_set_animation_globals(struct AnimInfo *node, s32 hasAnimation) {
    struct Animation *anim = node->curAnim;

    if (hasAnimation) {
        node->animFrame = geo_update_animation_frame(node, &node->animFrameAccelAssist);
    }
    node->animTimer = gAreaUpdateCounter;
    if (anim->flags & ANIM_FLAG_HOR_TRANS) {
        gCurrAnimType = ANIM_TYPE_VERTICAL_TRANSLATION;
    } else if (anim->flags & ANIM_FLAG_VERT_TRANS) {
        gCurrAnimType = ANIM_TYPE_LATERAL_TRANSLATION;
    } else if (anim->flags & ANIM_FLAG_6) {
        gCurrAnimType = ANIM_TYPE_NO_TRANSLATION;
    } else {
        gCurrAnimType = ANIM_TYPE_TRANSLATION;
    }

    gCurrAnimFrame = node->animFrame;
#ifdef HIGH_FPS_PC
    if (node->prevAnimPtr == anim && node->prevAnimID == node->animID &&
        gGlobalTimer == node->prevAnimFrameTimestamp + 1) {
        gPrevAnimFrame = node->prevAnimFrame;
    } else {
        gPrevAnimFrame = node->animFrame;
    }
    node->prevAnimPtr = anim;
    node->prevAnimID = node->animID;
    node->prevAnimFrame = node->animFrame;
    node->prevAnimFrameTimestamp = gGlobalTimer;
#endif

    gCurrAnimEnabled = (anim->flags & ANIM_FLAG_5) == 0;
    gCurrAnimAttribute = segmented_to_virtual((void *) anim->index);
    gCurrAnimData = segmented_to_virtual((void *) anim->values);

    if (anim->animYTransDivisor == 0) {
        gCurrAnimTranslationMultiplier = 1.0f;
    } else {
        gCurrAnimTranslationMultiplier = (f32) node->animYTrans / (f32) anim->animYTransDivisor;
    }
}

/**
 * Process a shadow node. Renders a shadow under an object offset by the
 * translation of the first animated component and rotated according to
 * the floor below it.
 */
static void geo_process_shadow(struct GraphNodeShadow *node) {
    Gfx *shadowList;
#ifdef HIGH_FPS_PC
    Gfx *shadowListInterpolated;
#endif
    Mat4 mtxf;
    Vec3f shadowPos;
#ifdef HIGH_FPS_PC
    Vec3f shadowPosInterpolated;
#endif
    Vec3f animOffset;
    f32 objScale;
    f32 shadowScale;
    f32 sinAng;
    f32 cosAng;
    struct GraphNode *geo;
    Mtx *mtx;
#ifdef HIGH_FPS_PC
    Mtx *mtxInterpolated;
#endif

    if (gCurGraphNodeCamera != NULL && gCurGraphNodeObject != NULL) {
        if (gCurGraphNodeHeldObject != NULL) {
            get_pos_from_transform_mtx(shadowPos, gMatStack[gMatStackIndex],
                                       *gCurGraphNodeCamera->matrixPtr);
            shadowScale = node->shadowScale;
        } else {
            vec3f_copy(shadowPos, gCurGraphNodeObject->pos);
            shadowScale = node->shadowScale * gCurGraphNodeObject->scale[0];
        }

        objScale = 1.0f;
        if (gCurrAnimEnabled) {
            if (gCurrAnimType == ANIM_TYPE_TRANSLATION
                || gCurrAnimType == ANIM_TYPE_LATERAL_TRANSLATION) {
                geo = node->node.children;
                if (geo != NULL && geo->type == GRAPH_NODE_TYPE_SCALE) {
                    objScale = ((struct GraphNodeScale *) geo)->scale;
                }
                animOffset[0] =
                    gCurrAnimData[retrieve_animation_index(gCurrAnimFrame, &gCurrAnimAttribute)]
                    * gCurrAnimTranslationMultiplier * objScale;
                animOffset[1] = 0.0f;
                gCurrAnimAttribute += 2;
                animOffset[2] =
                    gCurrAnimData[retrieve_animation_index(gCurrAnimFrame, &gCurrAnimAttribute)]
                    * gCurrAnimTranslationMultiplier * objScale;
                gCurrAnimAttribute -= 6;

                // simple matrix rotation so the shadow offset rotates along with the object
                sinAng = sins(gCurGraphNodeObject->angle[1]);
                cosAng = coss(gCurGraphNodeObject->angle[1]);

                shadowPos[0] += animOffset[0] * cosAng + animOffset[2] * sinAng;
                shadowPos[2] += -animOffset[0] * sinAng + animOffset[2] * cosAng;
            }
        }

#ifdef HIGH_FPS_PC
        if (gCurGraphNodeHeldObject != NULL) {
            if (gGlobalTimer == gCurGraphNodeHeldObject->prevShadowPosTimestamp + 1) {
                interpolate_vectors(shadowPosInterpolated, gCurGraphNodeHeldObject->prevShadowPos, shadowPos);
            } else {
                vec3f_copy(shadowPosInterpolated, shadowPos);
            }
            vec3f_copy(gCurGraphNodeHeldObject->prevShadowPos, shadowPos);
            gCurGraphNodeHeldObject->prevShadowPosTimestamp = gGlobalTimer;
        } else {
            if (gGlobalTimer == gCurGraphNodeObject->prevShadowPosTimestamp + 1 &&
                gGlobalTimer != gCurGraphNodeObject->skipInterpolationTimestamp) {
                interpolate_vectors(shadowPosInterpolated, gCurGraphNodeObject->prevShadowPos, shadowPos);
            } else {
                vec3f_copy(shadowPosInterpolated, shadowPos);
            }
            vec3f_copy(gCurGraphNodeObject->prevShadowPos, shadowPos);
            gCurGraphNodeObject->prevShadowPosTimestamp = gGlobalTimer;
        }

        shadowListInterpolated = create_shadow_below_xyz(shadowPosInterpolated[0], shadowPosInterpolated[1],
                                                         shadowPosInterpolated[2], shadowScale,
                                                         node->shadowSolidity, node->shadowType);
#endif

        shadowList = create_shadow_below_xyz(shadowPos[0], shadowPos[1], shadowPos[2], shadowScale,
                                             node->shadowSolidity, node->shadowType);
#ifdef HIGH_FPS_PC
        if (shadowListInterpolated != NULL && shadowList != NULL)
#else
        if (shadowList != NULL)
#endif
        {
            mtx = alloc_display_list(sizeof(*mtx));
#ifdef HIGH_FPS_PC
            mtxInterpolated = alloc_display_list(sizeof(*mtxInterpolated));
#endif
            gMatStackIndex++;

            mtxf_translate(mtxf, shadowPos);
            mtxf_mul(gMatStack[gMatStackIndex], mtxf, *gCurGraphNodeCamera->matrixPtr);
            mtxf_to_mtx(mtx, gMatStack[gMatStackIndex]);
            gMatStackFixed[gMatStackIndex] = mtx;
            
#ifdef HIGH_FPS_PC
            mtxf_translate(mtxf, shadowPosInterpolated);
            mtxf_mul(gMatStackInterpolated[gMatStackIndex], mtxf, *gCurGraphNodeCamera->matrixPtrInterpolated);
            mtxf_to_mtx(mtxInterpolated, gMatStackInterpolated[gMatStackIndex]);
            gMatStackInterpolatedFixed[gMatStackIndex] = mtxInterpolated;
#endif

#ifdef HIGH_FPS_PC
            if (gShadowAboveWaterOrLava == TRUE) {
                geo_append_display_list2((void *) VIRTUAL_TO_PHYSICAL(shadowList),
                                         (void *) VIRTUAL_TO_PHYSICAL(shadowListInterpolated), 4);
            } else if (gMarioOnIceOrCarpet == 1) {
                geo_append_display_list2((void *) VIRTUAL_TO_PHYSICAL(shadowList),
                                         (void *) VIRTUAL_TO_PHYSICAL(shadowListInterpolated), 5);
            } else {
                geo_append_display_list2((void *) VIRTUAL_TO_PHYSICAL(shadowList),
                                         (void *) VIRTUAL_TO_PHYSICAL(shadowListInterpolated), 6);
            }
#else
            if (gShadowAboveWaterOrLava == TRUE) {
                geo_append_display_list((void *) VIRTUAL_TO_PHYSICAL(shadowList), 4);
            } else if (gMarioOnIceOrCarpet == 1) {
                geo_append_display_list((void *) VIRTUAL_TO_PHYSICAL(shadowList), 5);
            } else {
                geo_append_display_list((void *) VIRTUAL_TO_PHYSICAL(shadowList), 6);
            }
#endif
            gMatStackIndex--;
        }
    }
    if (node->node.children != NULL) {
        geo_process_node_and_siblings(node->node.children);
    }
}

/**
 * Check whether an object is in view to determine whether it should be drawn.
 * This is known as frustum culling.
 * It checks whether the object is far away, very close / behind the camera,
 * or horizontally out of view. It does not check whether it is vertically
 * out of view. It assumes a sphere of 300 units around the object's position
 * unless the object has a culling radius node that specifies otherwise.
 *
 * The matrix parameter should be the top of the matrix stack, which is the
 * object's transformation matrix times the camera 'look-at' matrix. The math
 * is counter-intuitive, but it checks column 3 (translation vector) of this
 * matrix to determine where the origin (0,0,0) in object space will be once
 * transformed to camera space (x+ = right, y+ = up, z = 'coming out the screen').
 * In 3D graphics, you typically model the world as being moved in front of a
 * static camera instead of a moving camera through a static world, which in
 * this case simplifies calculations. Note that the perspective matrix is not
 * on the matrix stack, so there are still calculations with the fov to compute
 * the slope of the lines of the frustum.
 *
 *        z-
 *
 *  \     |     /
 *   \    |    /
 *    \   |   /
 *     \  |  /
 *      \ | /
 *       \|/
 *        C       x+
 *
 * Since (0,0,0) is unaffected by rotation, columns 0, 1 and 2 are ignored.
 */
static s32 obj_is_in_view(struct GraphNodeObject *node, Mat4 matrix) {
    s16 cullingRadius;
    s16 halfFov; // half of the fov in in-game angle units instead of degrees
    struct GraphNode *geo;
    f32 hScreenEdge;

    if (node->node.flags & GRAPH_RENDER_INVISIBLE) {
        return FALSE;
    }

    geo = node->sharedChild;

    // ! @bug The aspect ratio is not accounted for. When the fov value is 45,
    // the horizontal effective fov is actually 60 degrees, so you can see objects
    // visibly pop in or out at the edge of the screen.
    halfFov = (gCurGraphNodeCamFrustum->fov / 2.0f + 1.0f) * 32768.0f / 180.0f + 0.5f;

    hScreenEdge = -matrix[3][2] * sins(halfFov) / coss(halfFov);
    // -matrix[3][2] is the depth, which gets multiplied by tan(halfFov) to get
    // the amount of units between the center of the screen and the horizontal edge
    // given the distance from the object to the camera.

    // This multiplication should really be performed on 4:3 as well,
    // but the issue will be more apparent on widescreen.
    hScreenEdge *= GFX_DIMENSIONS_ASPECT_RATIO;

    if (geo != NULL && geo->type == GRAPH_NODE_TYPE_CULLING_RADIUS) {
        cullingRadius =
            (f32)((struct GraphNodeCullingRadius *) geo)->cullingRadius; //! Why is there a f32 cast?
    } else {
        cullingRadius = 300;
    }

    // Don't render if the object is close to or behind the camera
    if (matrix[3][2] > -100.0f + cullingRadius) {
        return FALSE;
    }

    //! This makes the HOLP not update when the camera is far away, and it
    //  makes PU travel safe when the camera is locked on the main map.
    //  If Mario were rendered with a depth over 65536 it would cause overflow
    //  when converting the transformation matrix to a fixed point matrix.
    if (matrix[3][2] < -20000.0f - cullingRadius) {
        return FALSE;
    }

    // Check whether the object is horizontally in view
    if (matrix[3][0] > hScreenEdge + cullingRadius) {
        return FALSE;
    }
    if (matrix[3][0] < -hScreenEdge - cullingRadius) {
        return FALSE;
    }
    return TRUE;
}

#ifdef HIGH_FPS_PC
static void interpolate_matrix(Mat4 result, Mat4 a, Mat4 b) {
    s32 i, j;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            result[i][j] = (a[i][j] + b[i][j]) / 2.0f;
        }
    }
}
#endif

/**
 * Process an object node.
 */
static void geo_process_object(struct Object *node) {
    Mat4 mtxf;
    s32 hasAnimation = (node->header.gfx.node.flags & GRAPH_RENDER_HAS_ANIMATION) != 0;
    u8 i;

#ifdef HIGH_FPS_PC
    Vec3f scaleInterpolated;
#endif

    if (node->header.gfx.areaIndex == gCurGraphNodeRoot->areaIndex) {
        if (node->header.gfx.throwMatrix != NULL) {
            mtxf_mul(gMatStack[gMatStackIndex + 1], *node->header.gfx.throwMatrix,
                     gMatStack[gMatStackIndex]);
#ifdef HIGH_FPS_PC
            if (gGlobalTimer == node->header.gfx.prevThrowMatrixTimestamp + 1 &&
                gGlobalTimer != node->header.gfx.skipInterpolationTimestamp) {
                interpolate_matrix(mtxf, *node->header.gfx.throwMatrix, node->header.gfx.prevThrowMatrix);
                mtxf_mul(gMatStackInterpolated[gMatStackIndex + 1], mtxf,
                     gMatStackInterpolated[gMatStackIndex]);
            } else {
                mtxf_mul(gMatStackInterpolated[gMatStackIndex + 1], (void *) node->header.gfx.throwMatrix,
                         gMatStackInterpolated[gMatStackIndex]);
            }
            mtxf_copy(node->header.gfx.prevThrowMatrix, *node->header.gfx.throwMatrix);
            node->header.gfx.prevThrowMatrixTimestamp = gGlobalTimer;
#endif
        } else if (node->header.gfx.node.flags & GRAPH_RENDER_CYLBOARD) {
#ifdef HIGH_FPS_PC
            Vec3f posInterpolated;
            if (gGlobalTimer == node->header.gfx.prevTimestamp + 1 &&
                gGlobalTimer != node->header.gfx.skipInterpolationTimestamp) {
                interpolate_vectors(posInterpolated, node->header.gfx.prevPos, node->header.gfx.pos);
            } else {
                vec3f_copy(posInterpolated, node->header.gfx.pos);
            }
            vec3f_copy(node->header.gfx.prevPos, node->header.gfx.pos);
            node->header.gfx.prevTimestamp = gGlobalTimer;
#endif
            mtxf_cylboard(gMatStack[gMatStackIndex + 1], gMatStack[gMatStackIndex],
                           node->header.gfx.pos, gCurGraphNodeCamera->roll);
#ifdef HIGH_FPS_PC
            mtxf_cylboard(gMatStackInterpolated[gMatStackIndex + 1], gMatStackInterpolated[gMatStackIndex],
                           posInterpolated, gCurGraphNodeCamera->roll);
#endif
        } else if (node->header.gfx.node.flags & GRAPH_RENDER_BILLBOARD) {
#ifdef HIGH_FPS_PC
            Vec3f posInterpolated;
            if (gGlobalTimer == node->header.gfx.prevTimestamp + 1 &&
                gGlobalTimer != node->header.gfx.skipInterpolationTimestamp) {
                interpolate_vectors(posInterpolated, node->header.gfx.prevPos, node->header.gfx.pos);
            } else {
                vec3f_copy(posInterpolated, node->header.gfx.pos);
            }
            vec3f_copy(node->header.gfx.prevPos, node->header.gfx.pos);
            node->header.gfx.prevTimestamp = gGlobalTimer;
#endif
            mtxf_billboard(gMatStack[gMatStackIndex + 1], gMatStack[gMatStackIndex],
                           node->header.gfx.pos, gCurGraphNodeCamera->roll);
#ifdef HIGH_FPS_PC
            mtxf_billboard(gMatStackInterpolated[gMatStackIndex + 1], gMatStackInterpolated[gMatStackIndex],
                           posInterpolated, gCurGraphNodeCamera->roll);
#endif
        } else {
#ifdef HIGH_FPS_PC
            Vec3f posInterpolated;
            Vec3s angleInterpolated;
            if (gGlobalTimer == node->header.gfx.prevTimestamp + 1 &&
                gGlobalTimer != node->header.gfx.skipInterpolationTimestamp) {
                interpolate_vectors(posInterpolated, node->header.gfx.prevPos, node->header.gfx.pos);
                interpolate_angles(angleInterpolated, node->header.gfx.prevAngle, node->header.gfx.angle);
            } else {
                vec3f_copy(posInterpolated, node->header.gfx.pos);
                vec3s_copy(angleInterpolated, node->header.gfx.angle);
            }
            vec3f_copy(node->header.gfx.prevPos, node->header.gfx.pos);
            vec3s_copy(node->header.gfx.prevAngle, node->header.gfx.angle);
            node->header.gfx.prevTimestamp = gGlobalTimer;
#endif
            mtxf_rotate_zxy_and_translate(mtxf, node->header.gfx.pos, node->header.gfx.angle);
            mtxf_mul(gMatStack[gMatStackIndex + 1], mtxf, gMatStack[gMatStackIndex]);
#ifdef HIGH_FPS_PC
            mtxf_rotate_zxy_and_translate(mtxf, posInterpolated, angleInterpolated);
            mtxf_mul(gMatStackInterpolated[gMatStackIndex + 1], mtxf, gMatStackInterpolated[gMatStackIndex]);
#endif
        }

#ifdef HIGH_FPS_PC
        if (gGlobalTimer == node->header.gfx.prevScaleTimestamp + 1 &&
            gGlobalTimer != node->header.gfx.skipInterpolationTimestamp) {
            interpolate_vectors(scaleInterpolated, node->header.gfx.prevScale, node->header.gfx.scale);
        } else {
            vec3f_copy(scaleInterpolated, node->header.gfx.scale);
        }
        vec3f_copy(node->header.gfx.prevScale, node->header.gfx.scale);
        node->header.gfx.prevScaleTimestamp = gGlobalTimer;
#endif

        mtxf_scale_vec3f(gMatStack[gMatStackIndex + 1], gMatStack[gMatStackIndex + 1],
                         node->header.gfx.scale);
#ifdef HIGH_FPS_PC 
        mtxf_scale_vec3f(gMatStackInterpolated[gMatStackIndex + 1], gMatStackInterpolated[gMatStackIndex + 1],
                         scaleInterpolated);
#endif
        node->header.gfx.throwMatrix = &gMatStack[++gMatStackIndex];
#ifdef HIGH_FPS_PC
        node->header.gfx.throwMatrixInterpolated = &gMatStackInterpolated[gMatStackIndex];
#endif        
        node->header.gfx.cameraToObject[0] = gMatStack[gMatStackIndex][3][0];
        node->header.gfx.cameraToObject[1] = gMatStack[gMatStackIndex][3][1];
        node->header.gfx.cameraToObject[2] = gMatStack[gMatStackIndex][3][2];

        // FIXME: correct types
        if (node->header.gfx.animInfo.curAnim != NULL) {
            geo_set_animation_globals(&node->header.gfx.animInfo, hasAnimation);
        }
        if (obj_is_in_view(&node->header.gfx, gMatStack[gMatStackIndex])) {
            Mtx *mtx = alloc_display_list(sizeof(*mtx));
            
            // Create the displaylist to set the active point lights
            Gfx* pointLightsDl = createPointLightsDl(&node->oPosX, 80.0f);

            // Put the lights on every layer, this can be optimized in the future
            // It will require some geolayout command to specify which layers this object uses
            // Maybe this can be implemented in a GEO_ASM call, where the parameter is a layer mask
            for (i = LAYER_FORCE; i <= LAYER_TRANSPARENT_INTER; i++)
            {
                geo_append_display_list(pointLightsDl, i);
            }
#ifdef HIGH_FPS_PC
            Mtx *mtxInterpolated = alloc_display_list(sizeof(*mtxInterpolated));
#endif

            mtxf_to_mtx(mtx, gMatStack[gMatStackIndex]);
            gMatStackFixed[gMatStackIndex] = mtx;
#ifdef HIGH_FPS_PC
            mtxf_to_mtx(mtxInterpolated, gMatStackInterpolated[gMatStackIndex]);
            gMatStackInterpolatedFixed[gMatStackIndex] = mtxInterpolated;
#endif
            if (node->header.gfx.sharedChild != NULL) {
                gCurGraphNodeObject = (struct GraphNodeObject *) node;
                node->header.gfx.sharedChild->parent = &node->header.gfx.node;
                geo_process_node_and_siblings(node->header.gfx.sharedChild);
                node->header.gfx.sharedChild->parent = NULL;
                gCurGraphNodeObject = NULL;
            }
            if (node->header.gfx.node.children != NULL) {
                geo_process_node_and_siblings(node->header.gfx.node.children);
            }
        }
#ifdef HIGH_FPS_PC
        else {
            node->header.gfx.prevThrowMatrixTimestamp = 0;
            node->header.gfx.prevTimestamp = 0;
            node->header.gfx.prevScaleTimestamp = 0;
        }
#endif

        gMatStackIndex--;
        gCurrAnimType = ANIM_TYPE_NONE;
        node->header.gfx.throwMatrix = NULL;
#ifdef HIGH_FPS_PC
        node->header.gfx.throwMatrixInterpolated = NULL;
#endif
    }
}

/**
 * Process an object parent node. Temporarily assigns itself as the parent of
 * the subtree rooted at 'sharedChild' and processes the subtree, after which the
 * actual children are be processed. (in practice they are null though)
 */
static void geo_process_object_parent(struct GraphNodeObjectParent *node) {
    if (node->sharedChild != NULL) {
        node->sharedChild->parent = (struct GraphNode *) node;
        geo_process_node_and_siblings(node->sharedChild);
        node->sharedChild->parent = NULL;
    }
    if (node->node.children != NULL) {
        geo_process_node_and_siblings(node->node.children);
    }
}

/**
 * Process a held object node.
 */
void geo_process_held_object(struct GraphNodeHeldObject *node) {
    Mat4 mat;
    Vec3f translation;
    Mtx *mtx = alloc_display_list(sizeof(*mtx));
#ifdef HIGH_FPS_PC
    Mtx *mtxInterpolated = alloc_display_list(sizeof(*mtxInterpolated));
    Vec3f scaleInterpolated;
#endif

#ifdef F3DEX_GBI_2
    gSPLookAt(gDisplayListHead++, &lookAt);
#endif

    if (node->fnNode.func != NULL) {
        node->fnNode.func(GEO_CONTEXT_RENDER, &node->fnNode.node, gMatStack[gMatStackIndex]);
    }
    if (node->objNode != NULL && node->objNode->header.gfx.sharedChild != NULL) {
        s32 hasAnimation = (node->objNode->header.gfx.node.flags & GRAPH_RENDER_HAS_ANIMATION) != 0;

        translation[0] = node->translation[0] / 4.0f;
        translation[1] = node->translation[1] / 4.0f;
        translation[2] = node->translation[2] / 4.0f;

#ifdef HIGH_FPS_PC
        if (gGlobalTimer == node->objNode->header.gfx.prevScaleTimestamp + 1) {
            interpolate_vectors(scaleInterpolated, node->objNode->header.gfx.prevScale, node->objNode->header.gfx.scale);
        } else {
            vec3f_copy(scaleInterpolated, node->objNode->header.gfx.scale);
        }
        vec3f_copy(node->objNode->header.gfx.prevScale, node->objNode->header.gfx.scale);
        node->objNode->header.gfx.prevScaleTimestamp = gGlobalTimer;
#endif

        mtxf_translate(mat, translation);
        mtxf_copy(gMatStack[gMatStackIndex + 1], *gCurGraphNodeObject->throwMatrix);
        gMatStack[gMatStackIndex + 1][3][0] = gMatStack[gMatStackIndex][3][0];
        gMatStack[gMatStackIndex + 1][3][1] = gMatStack[gMatStackIndex][3][1];
        gMatStack[gMatStackIndex + 1][3][2] = gMatStack[gMatStackIndex][3][2];
        mtxf_mul(gMatStack[gMatStackIndex + 1], mat, gMatStack[gMatStackIndex + 1]);
        mtxf_scale_vec3f(gMatStack[gMatStackIndex + 1], gMatStack[gMatStackIndex + 1],
                         node->objNode->header.gfx.scale);
#ifdef HIGH_FPS_PC
        mtxf_copy(gMatStackInterpolated[gMatStackIndex + 1], (void *) gCurGraphNodeObject->throwMatrixInterpolated);
        gMatStackInterpolated[gMatStackIndex + 1][3][0] = gMatStackInterpolated[gMatStackIndex][3][0];
        gMatStackInterpolated[gMatStackIndex + 1][3][1] = gMatStackInterpolated[gMatStackIndex][3][1];
        gMatStackInterpolated[gMatStackIndex + 1][3][2] = gMatStackInterpolated[gMatStackIndex][3][2];
        mtxf_mul(gMatStackInterpolated[gMatStackIndex + 1], mat, gMatStackInterpolated[gMatStackIndex + 1]);
        mtxf_scale_vec3f(gMatStackInterpolated[gMatStackIndex + 1], gMatStackInterpolated[gMatStackIndex + 1],
                         scaleInterpolated);
#endif
        if (node->fnNode.func != NULL) {
            node->fnNode.func(GEO_CONTEXT_HELD_OBJ, &node->fnNode.node,
                              (struct AllocOnlyPool *) gMatStack[gMatStackIndex + 1]);
        }
        gMatStackIndex++;
        mtxf_to_mtx(mtx, gMatStack[gMatStackIndex]);
        gMatStackFixed[gMatStackIndex] = mtx;
#ifdef HIGH_FPS_PC
        mtxf_to_mtx(mtxInterpolated, gMatStackInterpolated[gMatStackIndex]);
        gMatStackInterpolatedFixed[gMatStackIndex] = mtxInterpolated;
#endif
        gGeoTempState.type = gCurrAnimType;
        gGeoTempState.enabled = gCurrAnimEnabled;
        gGeoTempState.frame = gCurrAnimFrame;
        gGeoTempState.translationMultiplier = gCurrAnimTranslationMultiplier;
        gGeoTempState.attribute = gCurrAnimAttribute;
        gGeoTempState.data = gCurrAnimData;
#ifdef HIGH_FPS_PC
        gGeoTempState.prevFrame = gPrevAnimFrame;
#endif
        gCurrAnimType = 0;
        gCurGraphNodeHeldObject = (void *) node;
        if (node->objNode->header.gfx.animInfo.curAnim != NULL) {
            geo_set_animation_globals(&node->objNode->header.gfx.animInfo, hasAnimation);
        }

        geo_process_node_and_siblings(node->objNode->header.gfx.sharedChild);
        gCurGraphNodeHeldObject = NULL;
        gCurrAnimType = gGeoTempState.type;
        gCurrAnimEnabled = gGeoTempState.enabled;
        gCurrAnimFrame = gGeoTempState.frame;
        gCurrAnimTranslationMultiplier = gGeoTempState.translationMultiplier;
        gCurrAnimAttribute = gGeoTempState.attribute;
        gCurrAnimData = gGeoTempState.data;
#ifdef HIGH_FPS_PC
        gPrevAnimFrame = gGeoTempState.prevFrame;
#endif
        gMatStackIndex--;
    }

    if (node->fnNode.node.children != NULL) {
        geo_process_node_and_siblings(node->fnNode.node.children);
    }
}

#include <stdio.h>

/**
 * Advanced lighting engine
 * Processes a scene light, setting its position and other properties
 */
void geo_process_scene_light(struct GraphNodeSceneLight *node)
{
    Vec3f pos;

    
    switch (node->lightType)
    {
        case LIGHT_TYPE_DIRECTIONAL:
            if (!gOverrideDirectionalLight)
            {
                // Set the directional light color
                gDirectionalLight.l->l.colc[0] = gDirectionalLight.l->l.col[0] = node->color[0];
                gDirectionalLight.l->l.colc[1] = gDirectionalLight.l->l.col[1] = node->color[1];
                gDirectionalLight.l->l.colc[2] = gDirectionalLight.l->l.col[2] = node->color[2];

                // Set the pre transformed light direction
                gLightDir[0] = node->a;
                gLightDir[1] = node->b;
                gLightDir[2] = node->c;
            }
            break;
        case LIGHT_TYPE_POINT:
        case LIGHT_TYPE_POINT_OCCLUDE:
            get_pos_from_transform_mtx(pos, gMatStack[gMatStackIndex],
                                    *gCurGraphNodeCamera->matrixPtr);
            // Set the given point light's color
            node->light->l.pl.colc[0] = node->light->l.pl.col[0] = node->color[0];
            node->light->l.pl.colc[1] = node->light->l.pl.col[1] = node->color[1];
            node->light->l.pl.colc[2] = node->light->l.pl.col[2] = node->color[2];

            // Floors, but is faster
            node->light->worldPos[0] = (s16)(s32)pos[0];
            node->light->worldPos[1] = (s16)(s32)pos[1];
            node->light->worldPos[2] = (s16)(s32)pos[2];

            // More accurate (rounding instead of flooring), but more costly
            //vec3f_to_vec3s(node->light->worldPos, pos);

            node->light->l.pl.quadratic_attenuation = node->a;
            node->light->l.pl.linear_attenuation = node->b;
            node->light->l.pl.constant_attenuation = (node->c == 0) ? 1 : node->c;
            break;
        case LIGHT_TYPE_AMBIENT:
            if (!gOverrideAmbientLight)
            {
                // Set the ambient light color
                gDirectionalLight.a.l.colc[0] = gDirectionalLight.a.l.col[0] = node->color[0];
                gDirectionalLight.a.l.colc[1] = gDirectionalLight.a.l.col[1] = node->color[1];
                gDirectionalLight.a.l.colc[2] = gDirectionalLight.a.l.col[2] = node->color[2];
            }
            break;
    }

    if (node->node.children != NULL) {
        geo_process_node_and_siblings(node->node.children);
    }
}

/**
 * Processes the children of the given GraphNode if it has any
 */
void geo_try_process_children(struct GraphNode *node) {
    if (node->children != NULL) {
        geo_process_node_and_siblings(node->children);
    }
}

/**
 * Process a generic geo node and its siblings.
 * The first argument is the start node, and all its siblings will
 * be iterated over.
 */
void geo_process_node_and_siblings(struct GraphNode *firstNode) {
    s16 iterateChildren = TRUE;
    struct GraphNode *curGraphNode = firstNode;
    struct GraphNode *parent = curGraphNode->parent;

    // In the case of a switch node, exactly one of the children of the node is
    // processed instead of all children like usual
    if (parent != NULL) {
        iterateChildren = (parent->type != GRAPH_NODE_TYPE_SWITCH_CASE);
    }

    do {
        if (curGraphNode->flags & GRAPH_RENDER_ACTIVE) {
            if (curGraphNode->flags & GRAPH_RENDER_CHILDREN_FIRST) {
                geo_try_process_children(curGraphNode);
            } else {
                switch (curGraphNode->type) {
                    case GRAPH_NODE_TYPE_ORTHO_PROJECTION:
                        geo_process_ortho_projection((struct GraphNodeOrthoProjection *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_PERSPECTIVE:
                        geo_process_perspective((struct GraphNodePerspective *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_MASTER_LIST:
                        geo_process_master_list((struct GraphNodeMasterList *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_LEVEL_OF_DETAIL:
                        geo_process_level_of_detail((struct GraphNodeLevelOfDetail *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_SWITCH_CASE:
                        geo_process_switch((struct GraphNodeSwitchCase *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_CAMERA:
                        geo_process_camera((struct GraphNodeCamera *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_TRANSLATION_ROTATION:
                        geo_process_translation_rotation(
                            (struct GraphNodeTranslationRotation *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_TRANSLATION:
                        geo_process_translation((struct GraphNodeTranslation *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_ROTATION:
                        geo_process_rotation((struct GraphNodeRotation *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_OBJECT:
                        geo_process_object((struct Object *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_ANIMATED_PART:
                        geo_process_animated_part((struct GraphNodeAnimatedPart *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_BILLBOARD:
                        geo_process_billboard((struct GraphNodeBillboard *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_DISPLAY_LIST:
                        geo_process_display_list((struct GraphNodeDisplayList *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_SCALE:
                        geo_process_scale((struct GraphNodeScale *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_SHADOW:
                        geo_process_shadow((struct GraphNodeShadow *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_OBJECT_PARENT:
                        geo_process_object_parent((struct GraphNodeObjectParent *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_GENERATED_LIST:
                        geo_process_generated_list((struct GraphNodeGenerated *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_BACKGROUND:
                        geo_process_background((struct GraphNodeBackground *) curGraphNode);
                        break;
                    case GRAPH_NODE_TYPE_HELD_OBJ:
                        geo_process_held_object((struct GraphNodeHeldObject *) curGraphNode);
                        break;
                    // Advanced lighting engine
                    case GRAPH_NODE_TYPE_SCENE_LIGHT:
                        geo_process_scene_light((struct GraphNodeSceneLight *) curGraphNode);
                        break;
                    default:
                        geo_try_process_children((struct GraphNode *) curGraphNode);
                        break;
                }
            }
        } else {
            if (curGraphNode->type == GRAPH_NODE_TYPE_OBJECT) {
                ((struct GraphNodeObject *) curGraphNode)->throwMatrix = NULL;
            }
        }
    } while (iterateChildren && (curGraphNode = curGraphNode->next) != firstNode);
}

/**
 * Process a root node. This is the entry point for processing the scene graph.
 * The root node itself sets up the viewport, then all its children are processed
 * to set up the projection and draw display lists.
 */
void geo_process_root(struct GraphNodeRoot *node, Vp *b, Vp *c, s32 clearColor) {
    UNUSED u8 filler[4];

    if (node->node.flags & GRAPH_RENDER_ACTIVE) {
        Mtx *initialMatrix;
        Vp *viewport = alloc_display_list(sizeof(*viewport));
#ifdef HIGH_FPS_PC
        Vp *viewportInterpolated = viewport;
#endif

#ifdef USE_SYSTEM_MALLOC
        gDisplayListHeap = alloc_only_pool_init();
#else
        gDisplayListHeap = alloc_only_pool_init(main_pool_available() - sizeof(struct AllocOnlyPool),
                                                MEMORY_POOL_LEFT);
#endif
        initialMatrix = alloc_display_list(sizeof(*initialMatrix));
        gMatStackIndex = 0;
        gCurrAnimType = 0;
        vec3s_set(viewport->vp.vtrans, node->x * 4, node->y * 4, 511);
        vec3s_set(viewport->vp.vscale, node->width * 4, node->height * 4, 511);
        if (b != NULL) {
            clear_framebuffer(clearColor);
#ifdef HIGH_FPS_PC
            viewportInterpolated = alloc_display_list(sizeof(*viewportInterpolated));
            interpolate_vectors_s16(viewportInterpolated->vp.vtrans, sPrevViewport.vp.vtrans, b->vp.vtrans);
            interpolate_vectors_s16(viewportInterpolated->vp.vscale, sPrevViewport.vp.vscale, b->vp.vscale);

            sViewportPos = gDisplayListHead;
            make_viewport_clip_rect(viewportInterpolated);
#else
            make_viewport_clip_rect(b);
#endif
            *viewport = *b;
        }

        else if (c != NULL) {
            clear_framebuffer(clearColor);
            make_viewport_clip_rect(c);
        }
#ifdef HIGH_FPS_PC
        sPrevViewport = *viewport;
#endif

        mtxf_identity(gMatStack[gMatStackIndex]);
        mtxf_to_mtx(initialMatrix, gMatStack[gMatStackIndex]);
        gMatStackFixed[gMatStackIndex] = initialMatrix;

#ifdef HIGH_FPS_PC
        mtxf_identity(gMatStackInterpolated[gMatStackIndex]);
        gMatStackInterpolatedFixed[gMatStackIndex] = initialMatrix;

        gSPViewport(gDisplayListHead++, VIRTUAL_TO_PHYSICAL(viewportInterpolated));
#else
        gSPViewport(gDisplayListHead++, VIRTUAL_TO_PHYSICAL(viewport));
#endif

        gSPMatrix(gDisplayListHead++, VIRTUAL_TO_PHYSICAL(gMatStackFixed[gMatStackIndex]),
                  G_MTX_MODELVIEW | G_MTX_LOAD | G_MTX_NOPUSH);
        gCurGraphNodeRoot = node;
        if (node->node.children != NULL) {
            geo_process_node_and_siblings(node->node.children);
        }
        gCurGraphNodeRoot = NULL;
        if (gShowDebugText) {
#ifndef USE_SYSTEM_MALLOC
            print_text_fmt_int(180, 36, "MEM %d",
                               gDisplayListHeap->totalSpace - gDisplayListHeap->usedSpace);
#endif
        }
        main_pool_free(gDisplayListHeap);
    }
}
