import java.io.File;

import org.bytedeco.javacpp.*;
import org.bytedeco.llvm.clang.*;
import org.bytedeco.llvm.global.clang;

/**
 * Reference
 * https://github.com/sabottenda/libclang-sample/blob/master/Diagnosis/Diagnosis.cc
 * 
 * @author Nico Hezel
 *
 */
public class ClangCodeCompleteExample {

	public static void main(String[] args) throws Exception {

		// current clang version
		System.out.println(clang.clang_getClangVersion().getString());

		// Location of the *.cc file
		File file = new File("sample3.cc");
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

		// Code Completion
		int lineno = 4;
		int columnno = 5;
		CXCodeCompleteResults compResults = clang.clang_codeCompleteAt(translationUnit, file.toString(), lineno, columnno, unsaved_files, 0, clang.clang_defaultCodeCompleteOptions());

		// show Completion results
		show_completion_results(compResults);
		
		// show Diagnosis
		show_diagnosis(translationUnit, compResults);

		// dispose all allocated data
		clang.clang_disposeCodeCompleteResults(compResults);
		clang.clang_disposeTranslationUnit(translationUnit);
		clang.clang_disposeIndex(index);

		System.out.println("finished");
	}

	protected static String getCompleteChunkKindSpelling(int chunkKind) {
		switch (chunkKind) {
		case clang.CXCompletionChunk_Optional:         return "Optional"; 
		case clang.CXCompletionChunk_TypedText:        return "TypedText"; 
		case clang.CXCompletionChunk_Text:             return "Text"; 
		case clang.CXCompletionChunk_Placeholder:      return "Placeholder";
		case clang.CXCompletionChunk_Informative:      return "Informative"; 
		case clang.CXCompletionChunk_CurrentParameter: return "CurrentParameter"; 
		case clang.CXCompletionChunk_LeftParen:        return "LeftParen"; 
		case clang.CXCompletionChunk_RightParen:       return "RightParen"; 
		case clang.CXCompletionChunk_LeftBracket:      return "LeftBracket"; 
		case clang.CXCompletionChunk_RightBracket:     return "RightBracket"; 
		case clang.CXCompletionChunk_LeftBrace:        return "LeftBrace"; 
		case clang.CXCompletionChunk_RightBrace:       return "RightBrace"; 
		case clang.CXCompletionChunk_LeftAngle:        return "LeftAngle"; 
		case clang.CXCompletionChunk_RightAngle:       return "RightAngle"; 
		case clang.CXCompletionChunk_Comma:            return "Comma"; 
		case clang.CXCompletionChunk_ResultType:       return "ResultType"; 
		case clang.CXCompletionChunk_Colon:            return "Colon"; 
		case clang.CXCompletionChunk_SemiColon:        return "SemiColon"; 
		case clang.CXCompletionChunk_Equal:            return "Equal"; 
		case clang.CXCompletionChunk_HorizontalSpace:  return "HorizontalSpace"; 
		case clang.CXCompletionChunk_VerticalSpace:    return "VerticalSpace"; 
		default:                                 	   return "Unknown"; 
		}
	}

	protected static String getCompletionAvailabilitySpelling(int availavility) {
		switch (availavility) {
		case clang.CXAvailability_Available:     return "Available"; 
		case clang.CXAvailability_Deprecated:    return "Deprecated"; 
		case clang.CXAvailability_NotAvailable:  return "NotAvailable"; 
		case clang.CXAvailability_NotAccessible: return "NotAccessible"; 
		default:                           		 return "Unknown"; 
		}
	}

	protected static String getKindTypeName(CXCursor cursor) {
		int curKind  = clang.clang_getCursorKind(cursor);
		String type;
		if (clang.clang_isAttribute(curKind) == 1) {
			type = "Attribute";
		} else if (clang.clang_isDeclaration(curKind) == 1) {
			type = "Declaration";
		} else if (clang.clang_isExpression(curKind) == 1) {
			type = "Expression";
		} else if (clang.clang_isInvalid(curKind) == 1) {
			type = "Invalid";
		} else if (clang.clang_isPreprocessing(curKind) == 1) {
			type = "Preprocessing";
		} else if (clang.clang_isReference(curKind) == 1) {
			type = "Reference";
		} else if (clang.clang_isStatement(curKind) == 1) {
			type = "Statement";
		} else if (clang.clang_isTranslationUnit(curKind) == 1) {
			type = "TranslationUnit";
		} else if (clang.clang_isUnexposed(curKind) == 1) {
			type = "Unexposed";
		} else {
			type = "Unknown";
		}
		return type;
	}

	protected static void show_completion_results(CXCodeCompleteResults compResults) {
		System.out.printf("=== show results ===\n");
		int[] isIncomplete = new int[1];
		int containerKind = clang.clang_codeCompleteGetContainerKind(compResults, isIncomplete);
		System.out.printf("Complete: %d\n", isIncomplete[0]);
		System.out.printf("Kind: %s\n", clang.clang_getCursorKindSpelling(containerKind).getString());
		System.out.printf("USR: %s\n", clang.clang_codeCompleteGetContainerUSR(compResults).getString());

		long context = clang.clang_codeCompleteGetContexts(compResults);
		System.out.printf("Context: %d\n", context);
		System.out.printf("\n");

		// show completion results
		System.out.printf("=== show completion results ===\n");
		System.out.printf("CodeCompleationResultsNum: %d\n", compResults.NumResults());
		for (int i = 0; i < compResults.NumResults(); i++) {
			System.out.printf("Results: %d\n", i);
			
			// TODO missing arrays access to different results
			CXCompletionResult result = compResults.Results().position(i);
			CXCompletionString compString = result.CompletionString();
			int kind = result.CursorKind();

			System.out.printf(" Kind: %s\n", clang.clang_getCursorKindSpelling(kind).getString());

			int availavility = clang.clang_getCompletionAvailability(compString);
			String availavilityText = getCompletionAvailabilitySpelling(availavility);
			System.out.printf(" Availavility: %s\n", availavilityText);

			int priority = clang.clang_getCompletionPriority(compString);
			System.out.printf(" Priority: %d\n", priority);

			CXString cxBriefComment = clang.clang_getCompletionBriefComment(compString);
			BytePointer cxBriefCommentStr = clang.clang_getCString(cxBriefComment);
			if(cxBriefCommentStr != null) {
				System.out.printf(" Comment: %s\n", cxBriefCommentStr.getString());
			}
			clang.clang_disposeString(cxBriefComment);

			int numChunks = clang.clang_getNumCompletionChunks(compString);
			System.out.printf(" NumChunks: %d\n", numChunks);
			for (int j = 0; j < numChunks; j++) {
				int chunkKind = clang.clang_getCompletionChunkKind(compString, j);
				System.out.printf("   Kind: %s Text: %s\n",
						getCompleteChunkKindSpelling(chunkKind),
						clang.clang_getCompletionChunkText(compString, j).getString());

				// TODO: check child chunks when CXCompletionChunk_Optional
				// CXCompletionString child = clang_getCompletionChunkCompletionString(compString);
			}

			int numAnnotations = clang.clang_getCompletionNumAnnotations(compString);
			System.out.printf(" NumAnnotation: %d\n", numAnnotations);
			for (int j = 0; j < numAnnotations; j++)
				System.out.printf("   Annotation: %s\n", clang.clang_getCompletionAnnotation(compString, j).getString());
			System.out.printf("\n");
		}
	}

	protected static void show_diagnosis(CXTranslationUnit tu, CXCodeCompleteResults compResults) {
		System.out.printf("=== show diagnosis ===\n");
		int numDiag = clang.clang_codeCompleteGetNumDiagnostics(compResults);
		System.out.printf("NumDiagnosis:%d\n", numDiag);
		for (int i = 0; i < numDiag; i++) {
			CXDiagnostic completeDiag = clang.clang_codeCompleteGetDiagnostic(compResults, i);
			System.out.printf(" Code Complete Diagnosis: %s\n", clang.clang_getDiagnosticSpelling(completeDiag).getString());
			
			CXDiagnostic diag = clang.clang_getDiagnostic(tu, i);
			System.out.printf(" Diagnosis: %s\n", clang.clang_getDiagnosticSpelling(diag).getString());
			
			clang.clang_disposeDiagnostic(completeDiag);
			clang.clang_disposeDiagnostic(diag);
		}
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
