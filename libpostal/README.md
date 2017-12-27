JavaCPP Presets for libpostal
========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * libpostal 1.0  https://github.com/openvenues/libpostal
 
libpostal is a C library for parsing/normalizing street addresses around the world using statistical NLP and open data.
The goal of this project is to understand location-based strings in every language, everywhere.

Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/libpostal/apidocs/


Sample Usage
------------
```java
String dataDir = "PATH_TO_LIBPOSTAL_TRAINING_DATA";
boolean setup3 = libpostal.libpostal_setup_datadir(dataDir);
boolean setup5 = libpostal.libpostal_setup_parser_datadir(dataDir);
boolean setup4 = libpostal.libpostal_setup_language_classifier_datadir(dataDir);

libpostal.libpostal_address_parser_options_t options = libpostal.libpostal_get_address_parser_default_options();
BytePointer address = new BytePointer("781 Franklin Ave Crown Heights Brooklyn NYC NY 11216 USA", StandardCharsets.UTF_8.name());
libpostal.libpostal_address_parser_response_t response = libpostal.libpostal_parse_address(address, options);
long count = response.num_components();
for (int i = 0; i < count; i++) {
    System.out.println(response.labels(i).getString() + " " + response.components(i).getString());
}

libpostal.libpostal_normalize_options_t normalizeOptions = libpostal.libpostal_get_default_options();
SizeTPointer sizeTPointer = new SizeTPointer(0);
address = new BytePointer("Quatre vingt douze Ave des Champs-Élysées", StandardCharsets.UTF_8.name());
PointerPointer result = libpostal.libpostal_expand_address(address, normalizeOptions, sizeTPointer);
long t_size = sizeTPointer.get(0);
for (long i = 0; i < t_size; i++) {
    System.out.println(result.getString(i));
}

libpostal.libpostal_teardown();
libpostal.libpostal_teardown_parser();
libpostal.libpostal_teardown_language_classifier();
System.exit(1);