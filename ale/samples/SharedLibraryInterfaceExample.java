/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare,
 *  Matthew Hausknecht, and the Reinforcement Learning and Artificial Intelligence 
 *  Laboratory
 * Released under the GNU General Public License; see License.txt for details. 
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  sharedLibraryInterfaceExample.cpp 
 *
 *  Sample code for running an agent with the shared library interface. 
 **************************************************************************** */

import java.lang.System;
import java.util.Random;
import org.bytedeco.javacpp.*;
import org.bytedeco.ale.*;
import static org.bytedeco.ale.global.ale.*;

public class SharedLibraryInterfaceExample {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java SharedLibraryInterfaceExample rom_file");
            System.exit(1);
        }

        ALEInterface ale = new ALEInterface();

        // Get & Set the desired settings
        ale.setInt("random_seed", 123);
        //The default is already 0.25, this is just an example
        ale.setFloat("repeat_action_probability", 0.25f);

        ale.setBool("display_screen", true);
        ale.setBool("sound", true);

        // Load the ROM file. (Also resets the system for new settings to
        // take effect.)
        ale.loadROM(args[0]);

        // Get the vector of legal actions
        IntPointer legal_actions = ale.getLegalActionSet();

        // Play 10 episodes
        Random random = new Random();
        for (int episode = 0; episode < 10; episode++) {
            float totalReward = 0;
            while (!ale.game_over()) {
                int a = legal_actions.get(random.nextInt((int)legal_actions.limit()));
                // Apply the action and get the resulting reward
                float reward = ale.act(a);
                totalReward += reward;
            }
            System.out.println("Episode " + episode + " ended with score: " + totalReward);
            ale.reset_game();
        }

        System.exit(0);
    }
}
