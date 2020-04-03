import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.bytedeco.javacpp.*;
import org.bytedeco.llvm.clang.*;
import org.bytedeco.llvm.global.clang;

/**
 * Reference
 * https://github.com/sabottenda/libclang-sample/blob/master/Token/Tokenize.cc
 * 
 * @author Nico Hezel
 *
 */
public class ClangTokenizeExample {

	public static void main(String[] args) throws Exception {

		// current clang version
		System.out.println(clang.clang_getClangVersion().getString());

		// Location of the *.cc file
		File file = new File("sample1.cc");
		System.out.println("Analyse file: "+ file); 

		// Command line parameters provided to clang.
		// https://clang.llvm.org/docs/ClangCommandLineReference.html
		String[] command_line_args = new String[] { };

		// Create an index containing all source files and essentially creating one executable
		// excludeDeclsFromPCH = 1, displayDiagnostics = 1
		CXIndex index = clang.clang_createIndex(1, 0);

		// Build a translation unit from the source file
		int num_unsaved_files = 0;
		CXUnsavedFile unsaved_files = new CXUnsavedFile();
		BytePointer source_filename = new BytePointer(file.toString());
		PointerPointer command_line_args_ptr = new PointerPointer(command_line_args);
		CXTranslationUnit translationUnit = new CXTranslationUnit();
		checkError(clang.clang_parseTranslationUnit2(index, source_filename, command_line_args_ptr, command_line_args.length,
				unsaved_files, num_unsaved_files, clang.CXTranslationUnit_None, translationUnit));

		// get CXSouceRange of the file
		CXSourceRange range = getFileRange(translationUnit, file.toString());

		// tokenize in the range		
		CXToken tokens = new CXToken();
		int[] numTokens = new int[1];
		clang.clang_tokenize(translationUnit, range, tokens, numTokens);
		
		// show tokens
		show_all_tokens(translationUnit, tokens, numTokens[0]);
	
		// dispose all allocated data
		clang.clang_disposeTranslationUnit(translationUnit);
		clang.clang_disposeIndex(index);
		clang.clang_disposeTokens(translationUnit, tokens, numTokens[0]);
	}


	protected static String getTokenKindSpelling(int kind) {
		switch (kind) {
		case clang.CXToken_Punctuation: return "Punctuation";
		case clang.CXToken_Keyword:     return "Keyword";
		case clang.CXToken_Identifier:  return "Identifier";
		case clang.CXToken_Literal:     return "Literal";
		case clang.CXToken_Comment:     return "Comment";
		default:                  		return "Unknown";
		}
	}
	
	protected static void show_location(CXTranslationUnit tu, CXToken token) {
		CXSourceLocation loc = clang.clang_getTokenLocation(tu, token);
		CXSourceRange range = clang.clang_getTokenExtent(tu, token);
		
		CXFile file = new CXFile();
		int[] line = new int[1], column = new int[1], offset = new int[1];
		clang.clang_getFileLocation(loc, file, line, column, offset);

		System.out.printf(" Text: %s\n", clang.clang_getTokenSpelling(tu, token).getString());
		System.out.printf(" Kind: %s\n", getTokenKindSpelling(clang.clang_getTokenKind(token)));
		System.out.printf(" Location: %s:%d:%d:%d\n",
				clang.clang_getFileName(file).getString(), line[0], column[0], offset[0]);
		System.out.printf(" Range: %d to %d\n", range.begin_int_data(), range.end_int_data());
		
		System.out.printf("\n");
	}

	protected static void show_all_tokens(CXTranslationUnit tu, CXToken tokens, int numTokens) {
		System.out.printf("=== show tokens ===\n");
		System.out.printf("NumTokens: %d\n", numTokens);
		for (int i = 0; i < numTokens; i++) {
			CXToken token = tokens.position(i);
			System.out.printf("Token: %d\n", i);
			show_location(tu, token);
		}
	}


	protected static CXSourceRange getFileRange(CXTranslationUnit tu, String filename) throws IOException {
		CXFile file = clang.clang_getFile(tu, filename);
		int fileSize = (int)Files.size(Paths.get(filename));

		// get top/last location of the file
		CXSourceLocation topLoc  = clang.clang_getLocationForOffset(tu, file, 0);
		CXSourceLocation lastLoc = clang.clang_getLocationForOffset(tu, file, fileSize);
		System.out.println("Tokenize for range "+topLoc.int_data()+" to "+lastLoc.int_data());
		if (clang.clang_equalLocations(topLoc,  clang.clang_getNullLocation()) == 1 ||
				clang.clang_equalLocations(lastLoc, clang.clang_getNullLocation())  == 1) {
			System.out.printf("cannot retrieve location\n");
			return null;
		}

		// make a range from locations
		CXSourceRange range = clang.clang_getRange(topLoc, lastLoc);
		if (clang.clang_Range_isNull(range) == 1) {
			System.out.printf("cannot retrieve range\n");
			return null;
		}

		return range;
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
