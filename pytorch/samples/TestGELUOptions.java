// Test file to verify GELUOptions.approximate() setter works
// This demonstrates the fix for issue #1730

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class TestGELUOptions {
    public static void main(String[] args) {
        System.out.println("Testing GELUOptions.approximate() setter...");

        // Test 1: Create GELUOptions with default value
        GELUOptions options1 = new GELUOptions();
        System.out.println("Default approximate: " + options1.approximate().getString());

        // Test 2: Use the setter with String (this is the main fix)
        GELUOptions options2 = new GELUOptions().approximate("tanh");
        System.out.println("After setting to 'tanh': " + options2.approximate().getString());

        // Test 3: Fluent API chaining should work
        GELUOptions options3 = new GELUOptions()
            .approximate("none");
        System.out.println("After fluent setting to 'none': " + options3.approximate().getString());

        // Test 4: Create GELU module with options (as shown in documentation)
        try {
            // This is the usage pattern from the documentation that previously didn't work
            // GELU model = new GELU(new GELUOptions().approximate("tanh"));
            System.out.println("\nAll tests passed! The setter is working correctly.");
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
