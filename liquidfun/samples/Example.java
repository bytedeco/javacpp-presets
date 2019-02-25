import org.bytedeco.javacpp.*;
import org.bytedeco.liquidfun.*;
import static org.bytedeco.liquidfun.global.liquidfun.*;

public class Example {
  public static void main(String[] args) {
    b2World w = new b2World(0.0f, -10.0f);
    b2BodyDef bd = new b2BodyDef();
    bd.type(b2_dynamicBody);
    bd.SetPosition(1.0f, 5.0f);
    b2CircleShape c = new b2CircleShape();
    c.m_radius(2.0f);
    b2FixtureDef fd = new b2FixtureDef();
    fd.shape(c).density(1.0f);
    b2Body b = w.CreateBody(bd);
    b.CreateFixture(fd);
    for (int i = 1; i <= 5; i++) {
      System.out.println(i + ": ball at " + b.GetPositionX() + "," + b.GetPositionY());
      w.Step(0.1f, 2, 8);
    }
    System.exit(0);
  }
}
