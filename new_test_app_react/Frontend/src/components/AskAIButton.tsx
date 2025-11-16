import { useAskAI } from "./useAskAI";
import { TooltipProvider, Tooltip as UITooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { Dialog, DialogContent, DialogOverlay, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { Button } from "@/components/ui/button";

const AskAIButton = ({ mainData, aiParams, parametersList }: { mainData: any, aiParams: any, parametersList: any[] }) => {
  const { aiModal, setAIModal, aiLoading, aiData, onOpenAI } = useAskAI(parametersList);

  function convertBoldMarkdownToHtml(text) {
    return text.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
  }

  return (
    <>
      <TooltipProvider>
         <UITooltip>
            <TooltipTrigger asChild>
            <div className="ml-auto">
                <Button
                onClick={() => onOpenAI(aiParams)}
                disabled={!(mainData?.length > 0)}
                variant="outline"
                size="icon"
                >
                <AutoAwesomeIcon className="h-5 w-5 text-muted-foreground hover:text-foreground" />
                </Button>
            </div>
            </TooltipTrigger>
            <TooltipContent>
            <p>Ask AI</p>
            </TooltipContent>
        </UITooltip>
      </TooltipProvider>

      <Dialog open={aiModal} onOpenChange={setAIModal}>
         <DialogOverlay className="bg-transparent fixed inset-0 z-40" />
        <DialogContent 
         className="bg-white border border-gray-300 text-black dark:bg-slate-800 dark:border-slate-600 dark:text-white w-full max-w-4xl max-h-[80vh] overflow-y-auto"
         onInteractOutside={(e) => e.preventDefault()}
        >
          <DialogHeader>
            <DialogTitle>Ask AI</DialogTitle>
          </DialogHeader>

          <div className="p-3 whitespace-pre-wrap min-h-[200px] flex items-center justify-center">
            {aiLoading ? (
              <div className="flex items-center justify-center w-full py-10">
                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-gray-500 dark:border-white"></div>
              </div>
            ) : (
              <div className="max-h-[60vh] overflow-y-auto"><div dangerouslySetInnerHTML={{ __html: convertBoldMarkdownToHtml(aiData) }} /></div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default AskAIButton;
