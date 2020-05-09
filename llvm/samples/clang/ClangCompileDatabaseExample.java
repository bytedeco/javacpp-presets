import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

import org.bytedeco.javacpp.*;
import org.bytedeco.llvm.clang.*;
import org.bytedeco.llvm.global.clang;

/**
 * Reference
 * http://bastian.rieck.ru/blog/posts/2016/baby_steps_libclang_function_extents/
 * 
 * @author Nico Hezel
 *
 */
public class ClangCompileDatabaseExample {

	public static void main(String[] args) throws Exception {

		// current clang version
		System.out.println(clang.clang_getClangVersion().getString());

		// prepare data: copy sample1.cc and create a compile_commands.json file in a temp dir
		Path tempDir = Files.createTempDirectory("clang");
		String cfileName = "sample2.cc";
		Path cFile = tempDir.resolve(cfileName);
		Files.copy(Paths.get(cfileName), cFile);

		// http://clang.llvm.org/docs/JSONCompilationDatabase.html
		String compile_commands = "[{"
				+ "  \"directory\": \""+tempDir.toString().replace('\\', '/')+"\",\r\n" 
				+ "  \"command\": \"\",\r\n" 
				+ "  \"file\": \""+cfileName+"\",\r\n" 
				+ "}]";
		Path commandsFile = Files.write(tempDir.resolve("compile_commands.json"), compile_commands.getBytes(StandardCharsets.US_ASCII));

		// Create an index containing all source files and essentially creating one executable
		// excludeDeclsFromPCH = 0, displayDiagnostics = 0
		CXIndex index = clang.clang_createIndex(1, 0);

		// Compile database from json file
		int[] errorCode = new int[1];
		CXCompilationDatabase compilationDatabase = clang.clang_CompilationDatabase_fromDirectory(tempDir.toString(), errorCode);
		switch (errorCode[0]) {
			case clang.CXCompilationDatabase_NoError:
				break;
			case clang.CXCompilationDatabase_CanNotLoadDatabase:
				throw new Exception("Cannot load database\n");
			default: 
				throw new Exception("unknown return\n");
		}

		// Print the compile commands of the database file
		CXCompileCommands compileCommands = clang.clang_CompilationDatabase_getAllCompileCommands(compilationDatabase);
		System.out.println("Obtained "+clang.clang_CompileCommands_getSize(compileCommands)+" compile commands");
		CXCompileCommand compileCommand = clang.clang_CompileCommands_getCommand(compileCommands, 0);
		int numArguments = clang.clang_CompileCommand_getNumArgs(compileCommand);
		String[] arguments = new String[numArguments];
		for(int i = 0; i < numArguments; i++)
			arguments[i] = clang.clang_CompileCommand_getArg(compileCommand, i).getString();
		System.out.println("Arguments :"+Arrays.toString(arguments));
		
		// use the compile commands and filename to analyze the content of the file
		String filename = clang.clang_CompileCommand_getFilename(compileCommand).getString();
		String dirname = clang.clang_CompileCommand_getDirectory(compileCommand).getString();
		String pathname = dirname +"/"+ filename;
		System.out.println("Analyse file: " + pathname);		

		// Build a translation unit of the source file
		int num_unsaved_files = 0;
		CXUnsavedFile unsaved_files = new CXUnsavedFile();
		BytePointer source_filename = new BytePointer(pathname);
		PointerPointer arguments_ptr = new PointerPointer(arguments);
		CXTranslationUnit translationUnit = new CXTranslationUnit();
		checkError(clang.clang_parseTranslationUnit2(index, source_filename, arguments_ptr, arguments.length,
				unsaved_files, num_unsaved_files, clang.CXTranslationUnit_None, translationUnit));

		// TODO do something with the data

		// Analyze the source code and print some basic diagnostic informations.
		int numDiagnostic = clang.clang_getNumDiagnostics(translationUnit);
		System.out.println("Found "+numDiagnostic+" diagnostic points");
		for (int i = 0; i != numDiagnostic; ++i) {
			CXDiagnostic unitDiagnostic = clang.clang_getDiagnostic(translationUnit, i);

			CXString diagnosticString = clang.clang_formatDiagnostic(unitDiagnostic, clang.clang_defaultDiagnosticDisplayOptions());
			BytePointer cString = clang.clang_getCString(diagnosticString);
			System.out.println(cString.getString());	

			clang.clang_disposeString(diagnosticString);
			clang.clang_disposeDiagnostic(unitDiagnostic);
		}

		// dispose all allocated data
		clang.clang_disposeTranslationUnit(translationUnit);
		clang.clang_CompileCommands_dispose(compileCommands);
		clang.clang_CompilationDatabase_dispose(compilationDatabase);
		clang.clang_disposeIndex(index);
		
		// remove files
		Files.delete(commandsFile);
		Files.delete(cFile);
		Files.delete(tempDir);
	}

	/**
	 * Check for errors of the compilation process.
	 * 
	 * @param errorCode
	 * @throws Exception
	 */
	protected static void checkError(int errorCode) throws Exception {
		if(errorCode != clang.CXError_Success) {
			switch (errorCode) {
				case clang.CXError_InvalidArguments:
					throw new Exception("InvalidArguments");
				case clang.CXError_ASTReadError:
					throw new Exception("ASTReadError");
				case clang.CXError_Crashed:
					throw new Exception("Crashed");
				case clang.CXError_Failure:
					throw new Exception("Failure");
			}
		}
	}
}
