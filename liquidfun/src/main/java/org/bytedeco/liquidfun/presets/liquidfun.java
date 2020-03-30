/*
 * Copyright (C) 2016-2020 Felix Andrews, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.liquidfun.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Felix Andrews
 */

@Properties(inherit = javacpp.class, target = "org.bytedeco.liquidfun", global = "org.bytedeco.liquidfun.global.liquidfun", value = {
    @Platform(value = {"linux-x86", "macosx", "windows-x86"},
              define = "LIQUIDFUN_EXTERNAL_LANGUAGE_API 1",
              include = {"<Box2D/Common/b2Settings.h>",
                         "<Box2D/Common/b2Math.h>",
                         "<Box2D/Common/b2Draw.h>",
                         "<Box2D/Common/b2IntrusiveList.h>",
                         "<Box2D/Collision/Shapes/b2Shape.h>",
                         "<Box2D/Collision/Shapes/b2CircleShape.h>",
                         "<Box2D/Collision/Shapes/b2EdgeShape.h>",
                         "<Box2D/Collision/Shapes/b2ChainShape.h>",
                         "<Box2D/Collision/Shapes/b2PolygonShape.h>",
                         "<Box2D/Collision/b2Collision.h>",
                         "<Box2D/Collision/b2BroadPhase.h>",
                         "<Box2D/Collision/b2Distance.h>",
                         "<Box2D/Collision/b2DynamicTree.h>",
                         "<Box2D/Collision/b2TimeOfImpact.h>",
                         "<Box2D/Dynamics/b2Body.h>",
                         "<Box2D/Dynamics/b2Fixture.h>",
                         "<Box2D/Dynamics/b2WorldCallbacks.h>",
                         "<Box2D/Dynamics/b2TimeStep.h>",
                         "<Box2D/Dynamics/b2World.h>",
                         "<Box2D/Dynamics/b2ContactManager.h>",
                         "<Box2D/Dynamics/Contacts/b2Contact.h>",
                         "<Box2D/Dynamics/Joints/b2Joint.h>",
                         "<Box2D/Dynamics/Joints/b2DistanceJoint.h>",
                         "<Box2D/Dynamics/Joints/b2FrictionJoint.h>",
                         "<Box2D/Dynamics/Joints/b2GearJoint.h>",
                         "<Box2D/Dynamics/Joints/b2MotorJoint.h>",
                         "<Box2D/Dynamics/Joints/b2MouseJoint.h>",
                         "<Box2D/Dynamics/Joints/b2PrismaticJoint.h>",
                         "<Box2D/Dynamics/Joints/b2PulleyJoint.h>",
                         "<Box2D/Dynamics/Joints/b2RevoluteJoint.h>",
                         "<Box2D/Dynamics/Joints/b2RopeJoint.h>",
                         "<Box2D/Dynamics/Joints/b2WeldJoint.h>",
                         "<Box2D/Dynamics/Joints/b2WheelJoint.h>",
                         "<Box2D/Particle/b2Particle.h>",
                         "<Box2D/Particle/b2ParticleGroup.h>",
                         "<Box2D/Particle/b2ParticleSystem.h>",
                         "liquidfun_adapters.h"
                         },
              link = "liquidfun@.2.3.0")
})
public class liquidfun implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "liquidfun"); }

    public void map(InfoMap infoMap) {
        infoMap
        .put(new Info("b2_maxFloat").skip())
        .put(new Info("b2_epsilon").skip())
        .put(new Info("b2_maxTranslationSquared").skip())
        .put(new Info("b2_maxRotationSquared").skip())
        .put(new Info("b2Inline").cppTypes().annotations().cppText(""))
        // template to specialize
        .put(new Info("b2TypedIntrusiveListNode<b2ParticleHandle>").pointerTypes("ParticleHandleListNode").define())
        // setters for const* members
        .put(new Info("b2FixtureDef::shape").javaText("@MemberGetter public native @Const b2Shape shape();\n" +
                                                      "@MemberSetter public native b2FixtureDef shape(@Const b2Shape shape);"))
        .put(new Info("b2ParticleGroupDef::shape").javaText("@MemberGetter public native @Const b2Shape shape();\n" +
                                                            "@MemberSetter public native b2ParticleGroupDef shape(@Const b2Shape shape);"))
        .put(new Info("b2ParticleGroupDef::shapes").javaText("@MemberGetter public native @Cast(\"const b2Shape*const*\") PointerPointer shapes();\n" +
                                                             "@MemberSetter public native b2ParticleGroupDef shapes(@Cast(\"const b2Shape*const*\") PointerPointer shapes);"))
        .put(new Info("b2ParticleGroupDef::positionData").javaText("@MemberGetter public native @Const b2Vec2 positionData();\n" +
                                                                   "@MemberSetter public native b2ParticleGroupDef positionData(@Const b2Vec2 positionData);"))
        // Java lacks C++'s unsigned ints, so promote to larger types to fit
        .put(new Info("uint8").cast().valueTypes("short").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
        .put(new Info("uint16").cast().valueTypes("int").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
        .put(new Info("uint32").cast().valueTypes("long").pointerTypes("IntPointer", "IntBuffer", "int[]"))
        // enable callbacks
        .put(new Info("b2QueryCallback", "b2RayCastCallback").virtualize())
        .put(new Info("b2ContactListener", "b2ContactFilter", "b2DestructionListener").virtualize())
        .put(new Info("b2Draw", "b2DynamicTreeQueryCallback", "b2DynamicTreeRayCastCallback").virtualize())
        .put(new Info("b2DynamicTree::Query<b2DynamicTreeQueryCallback>").javaNames("Query").define())
        .put(new Info("b2DynamicTree::RayCast<b2DynamicTreeRayCastCallback>").javaNames("RayCast").define())
        ;
    }
}
