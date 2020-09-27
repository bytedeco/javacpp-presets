/*
 * Copyright (C) 2011  Statoil ASA, Norway.
 * The file 'kw_list.c' is part of ERT - Ensemble based Reservoir Tool.
 * ERT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * ERT is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
 * for more details.
 */
import org.bytedeco.libecl.fortio_type;
import org.bytedeco.libecl.ecl_kw_type;
import static org.bytedeco.libecl.global.libecl.*;

public class KeywordsList {

    public static void main(String[] args) {
        for (String arg : args) {
            kw_list(arg);
        }
    }

    private static void kw_list(String filename) {
        fortio_type fortio;
        ecl_kw_type ecl_kw = ecl_kw_alloc_empty();
        boolean[] fmt_file = new boolean[1];
        if (ecl_util_fmt_file(filename, fmt_file)) {

            System.out.println("-----------------------------------------------------------------");
            System.out.printf("%s: %n", filename);
            boolean endian_flip_header = false;
            fortio = fortio_open_reader(filename, fmt_file[0], endian_flip_header);
            while (ecl_kw_fread_realloc(ecl_kw, fortio)) {
                ecl_kw_summarize(ecl_kw);
            }
            System.out.println("-----------------------------------------------------------------");

            ecl_kw_free(ecl_kw);
            fortio_fclose(fortio);
        } else {
            System.err.printf("Could not determine formatted/unformatted status of:%s - skipping%n", filename);
        }
    }

}
